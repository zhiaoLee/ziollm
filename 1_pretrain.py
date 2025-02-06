import argparse
import os
import math
import time
import pandas as pd
import torch
from torch import optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, DataLoader

from contextlib import nullcontext
from transformers import AutoTokenizer

from model.LMConfig import LMConfig
from model.model import Transformer
from model.dataset import PretrainDataset


def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)

# 定义学习率调度函数，根据当前迭代次数计算学习率，采用余弦退火策略
# 余弦退火算法的核心思想是根据训练的轮数动态地调整学习率，在训练初期使用较大的学习率，以便快速收敛到最优解附近；
# 随着训练的进行，逐渐减小学习率，使得模型能够在最优解附近进行更精细的调整，从而提高模型的性能。
def get_lr(it, all):
    warmup_iters = args.warmup_iters  #预热迭代次数
    lr_decay_iters = all  # 学习率衰减的总迭代次数
    min_lr = args.learning_rate / 10  # 最小学习率

    # 如果当前迭代次数小于预热迭代次数，使用线性预热策略
    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    # 如果当前迭代次数大于衰减迭代次数，返回最小学习率
    if it > lr_decay_iters:
        return min_lr
    # 计算衰减系数，使用余弦退火策略
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    # 余弦退火系数（cosine annealing coefficient）的计算
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (args.learning_rate - min_lr)


def train_epoch(epoch, wandb):
    start_time = time.time()
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr # 设置优化器的学习率

        with ctx:  # 使用混合精度训练（如果设备是 GPU）
            out = model(X, Y)
            loss = out.last_loss / args.accumulation_steps # 计算损失，并进行梯度累积
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()

        scaler.scale(loss).backward()  # 反向传播，计算梯度

        # 每 accumulation_steps 步进行一次梯度更新
        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)  # 反缩放梯度，unscale_ 方法的作用是将之前通过 scaler.scale(loss) 放大的梯度还原到原始大小，这样才能正确地进行梯度裁剪和参数更新。

            # 梯度裁剪是一种防止梯度爆炸的技术，通过限制梯度的范数（norm）来确保梯度不会变得过大。
            # torch.nn.utils.clip_grad_norm 函数会计算模型所有参数的梯度的范数，并将其裁剪到不超过 args.grad_clip 指定的阈值。
            torch.nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)  # 梯度裁剪

            scaler.step(optimizer) # 更新模型参数，根据反缩放后的梯度来更新模型的参数
            scaler.update() # 更新缩放器，根据当前梯度的情况自适应地调整缩放因子

            optimizer.zero_grad(set_to_none=True) # 清空梯度

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min:'.format(
                    epoch,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        # 每 args.save_interval 步保存一次模型
        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval() # 切换到评估模式
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.dim}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict() # 获取模型状态字典
            else:
                state_dict = model.state_dict()

            torch.save(state_dict, ckp)
            model.train() # 切换回训练模式


def init_model():
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    tokenizer = AutoTokenizer.from_pretrained('./model/ziollm_tokenizer')

    model = Transformer(lm_config).to(args.device)

    Logger(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    return model, tokenizer

def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    #初始化DDP，使用默认backend(nccl)就行。如果是CPU模型运行，需要选择其他后端。
    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    #根据local_rank来设定当前使用哪块GPU
    torch.cuda_set_device(DEVICE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ziollm Pretraining")
    parser.add_argument("--out_dir", type=str, default="out", help="Output directory")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type")
    parser.add_argument("--use_wandb", default=False, help="Use Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="ziollm-Pretrain", help="Weights & Biases project name")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for data loading")
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_data.csv", help="Path to training data")
    parser.add_argument("--ddp", action="store_true", help="Use DistributedDataParallel")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="Gradient accumulation steps") #梯度积累步骤
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping threshold") #梯度裁剪阈值
    parser.add_argument("--warmup_iters", type=int, default=0, help="Number of warmup iterations")
    parser.add_argument("--log_interval", type=int, default=100, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=1000, help="Model saving interval")
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank for distributed training')

    args = parser.parse_args()

    lm_config = LMConfig()
    max_seq_len = lm_config.max_seq_len
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * max_seq_len
    torch.manual_seed(1337)
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"ziollm-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # 使用torch.cuda.amp.autocast()上下文管理器包装模型的前向传递和损失计算
    ctx = nullcontext() if device_type=="cpu" else torch.cuda.amp.autocast()

    #
    ddp = int(os.environ.get("RANK", -1)) != -1
    ddp_local_rank, DEVICE = 0, "cuda:0"
    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, tokenizer = init_model()
    df = pd.read_csv(args.data_path)
    #df.sample() 方法用于从 DataFrame 中随机抽取行。frac=1.0 表示抽取的比例为 100%，
    # 也就是说，它会对 DataFrame 中的所有行进行随机重排，返回一个新的 DataFrame，其中包含原 DataFrame 的所有行，但行的顺序是随机的。
    df = df.sample(frac=1.0)
    train_ds = PretrainDataset(df, tokenizer, max_length=max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,  #当将其设置为 True 时，DataLoader 会将数据样本在返回之前预先分配到固定内存（页锁定内存，pinned memory）中。对于使用 GPU 进行计算的场景，固定内存的好处在于可以显著加快数据从 CPU 到 GPU 的传输速度。
        drop_last=False,  #若 drop_last 设为 True，那么最后一个不完整的批次就会被丢弃。若设为 False，则最后一个批次会包含剩余的所有样本，该批次的大小会小于设定的 batch_size。
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    # 混合精度训练。GradScaler 主要用于在混合精度训练过程中自动进行梯度缩放，以解决半精度浮点数表示范围有限导致的梯度下溢问题。
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if ddp:
        model._ddp_params_and_buffers_to_ignore = {"poc_cis"}
        model = DistributedDataParallel(model, device_ids=[ddp_local_rank])

    iter_per_epoch = len(train_loader)
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)












