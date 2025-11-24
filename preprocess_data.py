import os
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import pandas as pd # 用于处理parquet文件，尽管load_dataset通常会处理
import pyarrow # datasets库读取parquet文件需要

# ==============================================================================
# 0. 配置参数 (与训练脚本保持一致)
# ==============================================================================
class Config:
    # 模型名称或路径
    model_name_or_path = "/hy-tmp/Qwen-7B-Chat"
    # 数据集文件路径
    dataset_file_path = "/hy-tmp/llama2_medical_meadow_wikidoc_instruct_dataset_files/data/train-00000-of-00001.parquet"
    # 最大序列长度
    max_seq_length = 2048

config = Config()

# ==============================================================================
# 1. 加载分词器
# ==============================================================================
print(f"--- 步骤 1: 正在加载分词器，模型路径为: {config.model_name_or_path} ---")
tokenizer = AutoTokenizer.from_pretrained(
    config.model_name_or_path,
    trust_remote_code=True
)

# 如果pad_token未设置，手动设置为Qwen的<|endoftext|>
if tokenizer.pad_token is None:
    tokenizer.pad_token = '<|endoftext|>'
    tokenizer.pad_token_id = 151643
    print(f"分词器的pad_token未设置，已手动设置为: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
else:
    print(f"分词器的pad_token已设置: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

# 如果eos_token未设置，手动设置为Qwen的<|endoftext|>
if tokenizer.eos_token is None:
    tokenizer.eos_token = '<|endoftext|>'
    tokenizer.eos_token_id = 151643
    print(f"分词器的eos_token未设置，已手动设置为: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
else:
    print(f"分词器的eos_token已设置: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

# 设置Qwen的聊天模板
tokenizer.chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}"
    "{{ '<|im_start|>system\n' + message['content'] + '<|im_end|>\n' }}"
    "{% elif message['role'] == 'user' %}"
    "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|im_start|>assistant\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endif %}"
    "{% endfor %}"
)
print("Qwen的聊天模板已设置。")

print("-" * 50)

# ==============================================================================
# 2. 加载原始数据集
# ==============================================================================
print(f"--- 步骤 2: 正在从 {config.dataset_file_path} 加载原始数据集 ---")
raw_dataset = load_dataset("parquet", data_files=config.dataset_file_path)

# 检查数据集是否包含'train'分割
if 'train' not in raw_dataset:
    raise ValueError("数据集不包含 'train' 分割。请检查数据集结构。")

print("原始数据集加载成功:")
print(raw_dataset)

# 打印一个原始数据示例
print("\n--- 原始数据示例 (训练集第一条) ---")
raw_example = raw_dataset['train'][0]
print(raw_example)
print("-" * 50)

# ==============================================================================
# 3. 格式化为Qwen聊天模板
# ==============================================================================
print("--- 步骤 3: 正在将数据格式化为Qwen聊天模板 ---")

def format_qwen_chat(example):
    # 优先使用'prompt'作为用户输入，'output'作为助手的回答
    # 如果'prompt'不存在，则尝试组合'instruction'和'input'
    user_message = example['prompt'] if 'prompt' in example and example['prompt'] else \
                   (f"{example['instruction']}\n{example['input']}" if 'input' in example and example['input'] else example['instruction'])

    assistant_message = example['output']

    # 构建Qwen的对话格式
    messages = [
        {"role": "system", "content": "你是一个乐于助人的医疗助手。"}, # 可自定义系统提示
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message}
    ]

    # 使用分词器的apply_chat_template生成训练所需的字符串
    # add_generation_prompt=False 是为了在训练时，不自动添加最后一个<|im_start|>assistant\n
    # 而是让模型去生成它和后续的回答
    # tokenize=False 表示只返回字符串，不进行分词
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": formatted_text}

# 应用格式化函数到数据集
# 仅对一小部分数据进行map操作以快速查看效果，避免处理整个大文件
# 或者直接对 raw_example 进行测试
formatted_example = format_qwen_chat(raw_example)
print("\n--- 格式化后的数据示例 (第一条) ---")
print(formatted_example)
print("-" * 50)

# ==============================================================================
# 4. 分词 (Tokenization)
# ==============================================================================
print("--- 步骤 4: 正在对格式化后的数据进行分词 ---")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        max_length=config.max_seq_length,
        truncation=True,
        padding="max_length" # 填充到max_seq_length
    )

# 对格式化后的示例进行分词
tokenized_example = tokenize_function(formatted_example)

print("\n--- 分词后的数据示例 (第一条) ---")
print(f"输入ID (前50个): {tokenized_example['input_ids'][:50]}...")
print(f"注意力掩码 (前50个): {tokenized_example['attention_mask'][:50]}...")
print(f"输入ID的长度: {len(tokenized_example['input_ids'])}")

# 验证分词结果：将token ID解码回文本
decoded_text = tokenizer.decode(tokenized_example['input_ids'], skip_special_tokens=True)
print("\n--- 解码后的分词文本 (第一条) ---")
print(decoded_text)
print("-" * 50)

print("\n数据预处理演示完成。")
print("现在你可以看到原始数据是如何转换为适合Qwen训练的格式的。")