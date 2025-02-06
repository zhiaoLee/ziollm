import json
import os

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders


def train_tokenizer():
    def read_texts_from_jsonl(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                yield data['text']

    data_path = './dataset/tokenizer_train.jsonl'
    # 实例化一个BPE 分词模型
    tokenizer = Tokenizer(models.BPE())
    # 创建一个字节级别的预分词器，作用是在正式使用分词模型（如 BPE、WordPiece 等）进行分词之前，对输入文本进行一些初步的处理。
    # ByteLevel 表示该预分词器基于字节级别进行操作，而 add_prefix_space=False 是一个参数设置，用于控制是否在文本前添加一个空格。
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    special_tokens = ["<unk>", "<s>", "</s>"]

    # BPE 是一种子词分词算法，trainers.BpeTrainer 就是专门用来在给定的语料库上训练 BPE 分词模型的工具。
    # 通过不断合并最频繁出现的字符对，逐步构建出一个合适的词表，从而实现将文本分割成有意义的子词单元的目的。
    # 训练好的 BPE 分词器可以应用于自然语言处理的多个任务，如机器翻译、文本生成、问答系统等。
    trainer = trainers.BpeTrainer(
        vocab_size = 6400,
        special_tokens = special_tokens,
        show_progress = True,
        initial_alphabet = pre_tokenizers.ByteLevel.alphabet()
    )

    texts = read_texts_from_jsonl(data_path)
    tokenizer.train_from_iterator(texts, trainer)
    # 字节级别的解码器主要负责将经过字节级别编码（例如通过字节对编码BPE算法进行分词后得到的字节级别的token序列）后的 token 序列重新转换回原始的文本字符串。
    # 在自然语言处理流程中，分词器将文本拆分成 token，而解码器则是这个过程的逆操作，将token还原成原始文本，以便于人类阅读或者进行后续的文本处理。
    tokenizer.decoder = decoders.ByteLevel()
    assert tokenizer.token_to_id("<unk>") == 0
    assert tokenizer.token_to_id("<s>") == 1
    assert tokenizer.token_to_id("</s>") == 2

    tokenizer_dir = "./model/ziollm_tokenizer"
    os.makedirs(tokenizer_dir, exist_ok=True)
    tokenizer.save(os.path.join(tokenizer_dir, "tokenizer.json"))
    tokenizer.model.save("./model/ziollm_tokenizer")

    # 手动创建配置文件
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": True,
        "added_tokens_decoder": {
            "0": {
                "content": "<unk>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "1": {
                "content": "<s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            },
            "2": {
                "content": "</s>",
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        },
        "additional_special_tokens": [],
        "bos_token": "<s>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "</s>",
        "legacy": True,
        "model_max_length": 1000000000000000019884624838656,
        "pad_token": None,
        "sp_model_kwargs": {},
        "spaces_between_special_tokens": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<unk>",
        "use_default_system_prompt": False,
        "chat_template": "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% endif %}{% if system_message is defined %}{{ system_message }}{% endif %}{% for message in messages %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{{ '<s>user\\n' + content + '</s>\\n<s>assistant\\n' }}{% elif message['role'] == 'assistant' %}{{ content + '</s>' + '\\n' }}{% endif %}{% endfor %}"
    }

    with open(os.path.join(tokenizer_dir, "tokenizer_config.json"), "w", encoding="utf-8") as config_file:
        json.dump(config, config_file, ensure_ascii=False, indent=4)

    print("Tokenizer training completed and saved")


def main():
    train_tokenizer()


if __name__ == '__main__':
    main()