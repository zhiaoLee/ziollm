import math

import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Callable, List, Union

from torch.nn.modules.module import T
from transformers import PreTrainedModel, GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from .LMConfig import LMConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_pos_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    poc_cis = torch.polar(torch.ones_like(freqs), freqs) #complex64
    return poc_cis


def apply_rotary_emb(xq, xk, poc_cis):
    def unite_shape(poc_cis, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert poc_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return poc_cis.view(*shape)

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    poc_cis = unite_shape(poc_cis, xq_)
    # flatten 函数用于将输入的张量展平，即将一个多维张量转换为一维张量。flatten(3)表示从第 3 维度开始将后面的维度展平。
    # torch.view_as_complex(...)：将乘法结果转换为复数张量。前提是 xq_ * poc_cis 的结果张量的最后一个维度大小为 2，
    # 因为 torch.view_as_complex 要求输入张量的最后一个维度存储实部和虚部。
    xq_out = torch.view_as_real(xq_ * poc_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * poc_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class FeedForward(nn.Module):
    # 在前馈网络中，将中间层的维度设置为 multiple_of 的倍数可以优化线性层的性能。
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

# repeat_kv 是一个与多头注意力机制相关的优化技巧，主要用于减少内存使用和计算量
# 并非为每个头都单独计算和存储键值对，而是让多个头共享相同的键值对，也就是对键值对进行重复使用。
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args: LMConfig):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        assert args.n_heads % self.n_kv_heads == 0
        self.n_local_heads = args.n_heads  #???
        self.n_local_kv_heads = self.n_kv_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)

        ## wo：多头注意力机制中引入的概念，为了整合多头信息并进行线性变换
        # wo矩阵的主要作用之一就是将所有头的输出拼接起来后进行线性变换，把各个头提取的分散信息融合成一个统一的表示。
        # 具体来说，多头注意力机制会先将每个头的输出按照最后一个维度进行拼接，形成一个高维的张量，然后通过与矩阵相乘，
        # 将这个高维张量映射到一个与输入维度相同的空间中，得到最终的多头注意力输出。
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.k_cache, self.v_cache = None, None
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)

        self.dropout = args.dropout

        mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)

        # 在 PyTorch 里，当创建一个自定义的神经网络模块（继承自 torch.nn.Module）时，有时候需要存储一些不需要进行反向传播更新的张量，
        # 例如掩码（mask）、统计信息等。register_buffer 方法就可以将这些张量注册为模块的缓冲区，这样它们就会成为模块状态的一部分，
        # 可以随着模块一起保存和加载，但是不会被当作模型的参数（即不会在优化器更新参数时被更新）。
        self.register_buffer("mask", mask, persistent=False) # ？？？

    def forward(self, x: torch.Tensor, pos_cis: torch.Tensor, kv_cache=False):

        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, pos_cis)

        xk = repeat_kv(xk, self.n_rep)    ##???
        xv = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        ## 注意力分数和权重
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = scores + self.mask[:, :, :seqlen, :seqlen]

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.attn_dropout(scores)
        output = torch.matmul(scores, xv)

        # 拼接多头输出
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        # 对拼接后的多头输出进行线性变换，得到最终的多头注意力输出。
        output = self.wo(output)
        # 正则化作用
        output = self.resid_dropout(output)

        return output





class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: LMConfig):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        # self.head_dim = args.dim // args.n_heads    ## ???
        self.attention = Attention(args)

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)


        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,         ## ???
            dropout=args.dropout,
        )

    def forward(self, x, poc_cis, kv_cache=False):
        h = x + self.attention(self.attention_norm(x), poc_cis, kv_cache)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(PreTrainedModel):
    config_class = LMConfig
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: LMConfig = None):
        super().__init__(params)
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embedding = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)
        self.tok_embedding.weight = self.output.weight

        poc_cis = precompute_pos_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("pos_cis", poc_cis, persistent=False)

        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                # 正态分布：从给定均值和标准差的正态分布N(mean, std)中生成值，填充输入的张量或变量。
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * params.n_layers))

        self.last_loss = None
        # 通常用于因果语言模型（如GPT）。因果语言模型的任务是根据前文预测下一个词，
        # CausalLMOutputWithPast 会封装模型的输出结果，并且包含 past_key_values （键值缓存）信息，
        # 这对于加速文本生成过程非常重要。
        self.OUT = CausalLMOutputWithPast()

        self._no_split_modules = [name for name, _ in self.named_modules()]

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    """
    [KV Cache]
    在基于 Transformer 架构的模型（如 Llama）里，每一层的多头注意力机制会计算键（Key，K）、值（Value，V）和查询（Query，Q）。
    在文本生成任务中，模型会逐个生成 token，而对于已经处理过的 token，其对应的键和值在后续生成新 token 时不会发生变化。
    KV Cache 就是利用这一特性，将已经计算好的键和值缓存起来，避免在生成后续 token 时重复计算，从而提高生成效率。
    [工作流程]
    初始阶段：在生成第一个 token 时，模型正常计算所有 token（此时只有一个）的键和值，并将其存储在 KV Cache 中。
    后续生成阶段：当生成后续的 token 时，模型只需要计算当前新 token 的查询向量，然后利用 KV Cache 中已有的键和值进行注意力计算，而不需要重新计算之前token的键和值。
    这样，随着生成的 token 数量增加，计算量的增长主要集中在新 token 上，而不是所有已生成的 token。
    """
    def forward(self, tokens: Optional[torch.Tensor] = None, targets: Optional[torch.Tensor] = None,
                kv_cache=False, **kwargs):
        current_idx = 0
        if 'input_ids' in kwargs:
            tokens = kwargs['input_ids']
        if 'attention_mask' in kwargs:
            targets = kwargs['attention_mask']
        if 'current_idx' in kwargs:
            current_idx = int(kwargs['current_idx'])

        _bsz, seqlen = tokens.shape
        h = self.tok_embedding(tokens)
        h = self.dropout(h)
        poc_cis = self.pos_cis[current_idx:current_idx + seqlen]
        for idx, layer in enumerate(self.layers):
            h = layer(h, poc_cis, kv_cache)

        h = self.norm(h)

        if targets is not None:
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                             ignore_index=0, reduction='none')
        else:
            logits = self.output(h[:, [-1], :])
            self.last_loss = None

        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('last_loss', self.last_loss)

        return self.OUT

    ## @torch.inference_mode() 是 PyTorch 中的一个上下文管理器装饰器，
    # 用于在推理（即使用训练好的模型进行预测）阶段临时禁用梯度计算，以提高推理速度并减少内存消耗。
    @torch.inference_mode()
    def generate(self, idx, eos, max_new_tokens, temperature=0.7, top_k=8, stream=True, rp=1., kv_cache=True):
        index = idx.shape[1]
        init_inference = True
        while idx.shape[1] < max_new_tokens - 1:
            if init_inference or not kv_cache:
                inference_res, init_inference = self(idx, kv_cache=kv_cache), False
            else:
                inference_res = self(idx[:, -1:], kv_cache=kv_cache, current_idx=idx.shape[1] - 1)

            logits = inference_res.logits
            logits = logits[:, -1, :]

            if temperature == 0.0:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1, generator=None)

            if idx_next == eos:
                break

            idx = torch.cat((idx, idx_next), dim = 1)
            if stream:
                yield idx[:, index:]
        if not stream:
            yield idx[:, index:]

    @torch.inference_mode()
    def eval_answer(self, idx):
        idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
        inference_res = self(idx_cond)
        logits = inference_res.logits
        logits = logits[:, -1, :]
        return logits








