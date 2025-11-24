import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling # 用于处理数据批次
)
from peft import LoraConfig, get_peft_model # 移除了 prepare_model_for_kbit_training
from datasets import load_dataset
import bitsandbytes as bnb # 导入bnb以检查4bit线性层
import pandas as pd # 用于读取parquet文件
import pyarrow # datasets库读取parquet需要

# ==============================================================================
# 0. 配置参数
# ==============================================================================
class Config:
    # 模型配置
    model_name_or_path = "/hy-tmp/Qwen-7B-Chat" # 请确保这是你完整下载的模型路径

    # 数据集配置
    dataset_file_path = "/hy-tmp/llama2_medical_meadow_wikidoc_instruct_dataset_files/data/train-00000-of-00001.parquet"
    max_seq_length = 512 # 最大序列长度，根据你的显存和数据调整

    # LoRA配置
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    # Qwen模型常见的线性层名称，我们之前已经验证过
    lora_target_modules = ["c_attn", "c_proj", "w1", "w2"] # 之前自动提取的，这里直接使用

    # 量化和数据类型配置
    bf16 = True # 启用bfloat16计算，如果GPU支持 (RTX 30系列及以上通常支持)
    fp16 = False # 如果bf16=False，则设置为True

    # 训练配置
    output_dir = "./qwen_7b_medical_lora_output" # 模型保存路径
    num_train_epochs = 1 # 训练轮次
    per_device_train_batch_size = 1 # 每个设备的训练批次大小，10G显存通常只能是1
    gradient_accumulation_steps = 8 # 梯度累积步数，模拟更大的批次 (1 * 8 = 8)
    learning_rate = 2e-5 # 学习率
    logging_steps = 10 # 日志打印频率
    save_steps = 500 # 模型保存频率
    # eval_steps = 500 # 评估频率 (如果提供了验证集)
    # evaluation_strategy="steps", # 如果提供了验证集)
    warmup_ratio = 0.03 # 学习率预热比例
    weight_decay = 0.01 # 权重衰减
    gradient_checkpointing = True # 显存优化，强烈建议开启
    optim = "paged_adamw_8bit" # 8位优化器，Qlora推荐
    report_to = "none" # 不上报到wandb等，如果你需要可以配置
    remove_unused_columns = False # 避免移除模型可能需要的列

# 实例化配置
config = Config()

print("--- 阶段三: Qlora微调Qwen-7B-Chat (完整训练脚本) ---")
print(f"模型路径: {config.model_name_or_path}")
print(f"数据集路径: {config.dataset_file_path}")
print(f"LoRA Rank (r): {config.lora_r}, LoRA Alpha: {config.lora_alpha}")
print(f"训练轮次: {config.num_train_epochs}, 批次大小: {config.per_device_train_batch_size}, 梯度累积: {config.gradient_accumulation_steps}")
print(f"使用 BF16 计算: {config.bf16}, 使用 FP16 计算: {config.fp16}")
print("-" * 50)

# ==============================================================================
# 1. 加载分词器并配置
# ==============================================================================
print(f"--- 步骤 1: 正在加载分词器并配置 ---")
tokenizer = AutoTokenizer.from_pretrained(
    config.model_name_or_path,
    trust_remote_code=True
)

# 手动设置Qwen的pad_token和eos_token，以及chat_template
if tokenizer.pad_token is None:
    tokenizer.pad_token = '<|endoftext|>'
    tokenizer.pad_token_id = 151643
    print(f"分词器的pad_token未设置，已手动设置为: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
else:
    print(f"分词器的pad_token已设置: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

if tokenizer.eos_token is None:
    tokenizer.eos_token = '<|endoftext|>'
    tokenizer.eos_token_id = 151643
    print(f"分词器的eos_token未设置，已手动设置为: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
else:
    print(f"分词器的eos_token已设置: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

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
# 2. 加载和预处理数据集
# ==============================================================================
print(f"--- 步骤 2: 正在加载和预处理数据集 ---")
raw_dataset = load_dataset("parquet", data_files=config.dataset_file_path)

if 'train' not in raw_dataset:
    raise ValueError("数据集不包含 'train' 分割。请检查数据集结构。")

def format_qwen_chat(example):
    user_message = example['prompt'] if 'prompt' in example and example['prompt'] else \
                   (f"{example['instruction']}\n{example['input']}" if 'input' in example and example['input'] else example['instruction'])
    assistant_message = example['output']

    messages = [
        {"role": "system", "content": "你是一个乐于助人的医疗助手。"},
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message}
    ]
    formatted_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": formatted_text}

formatted_dataset = raw_dataset.map(
    format_qwen_chat,
    remove_columns=raw_dataset['train'].column_names,
    num_proc=os.cpu_count()
)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        max_length=config.max_seq_length,
        truncation=True,
        padding="max_length"
    )

tokenized_dataset = formatted_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
    num_proc=os.cpu_count()
)
print("数据集加载并预处理完成。")
print(f"训练集大小: {len(tokenized_dataset['train'])} 条")
print("-" * 50)

# ==============================================================================
# 3. 加载量化模型并配置LoRA适配器
# ==============================================================================
print(f"--- 步骤 3: 正在加载量化模型并配置LoRA适配器 ---")

# 配置4位量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
)

# 加载基础模型 (4位量化)
model = AutoModelForCausalLM.from_pretrained(
    config.model_name_or_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
)
print("基础模型已加载并量化。")

# 准备模型进行k-bit训练 (手动处理，避免嵌入层OOM)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
print("模型已为k-bit训练准备就绪 (手动配置)。")

# 定义LoRA配置
lora_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    lora_dropout=config.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=config.lora_target_modules,
)

# 将LoRA适配器添加到模型
model = get_peft_model(model, lora_config)
print("LoRA适配器已成功注入模型。")

# 打印可训练参数
model.print_trainable_parameters()
print("-" * 50)

# ==============================================================================
# 4. 配置训练参数并启动训练
# ==============================================================================
print("--- 步骤 4: 正在配置训练参数并启动训练 ---")

training_args = TrainingArguments(
    output_dir=config.output_dir,
    num_train_epochs=config.num_train_epochs,
    per_device_train_batch_size=config.per_device_train_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    learning_rate=config.learning_rate,
    logging_steps=config.logging_steps,
    save_steps=config.save_steps,
    warmup_ratio=config.warmup_ratio,
    weight_decay=config.weight_decay,
    fp16=config.fp16,
    bf16=config.bf16,
    gradient_checkpointing=config.gradient_checkpointing,
    optim=config.optim,
    report_to=config.report_to,
    remove_unused_columns=config.remove_unused_columns,
    # max_steps=-1 # 如果你不想用epochs，可以使用max_steps
    # 其他可能需要的参数，例如dataloader_num_workers等
)

# 数据收集器，用于将tokenized数据批处理并准备好送入模型
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 禁用缓存以节省显存 (对于Qlora训练通常不需要，但以防万一)
model.config.use_cache = False

# 开始训练
print("正在启动训练...")
trainer.train()

# ==============================================================================
# 5. 保存微调后的LoRA适配器和分词器
# ==============================================================================
print(f"\n--- 步骤 5: 正在保存微调后的LoRA适配器和分词器到 {config.output_dir}/final_checkpoint ---")
final_checkpoint_path = os.path.join(config.output_dir, "final_checkpoint")
trainer.save_model(final_checkpoint_path)
tokenizer.save_pretrained(final_checkpoint_path)

print("Qlora微调完成！")

# ==============================================================================
# 6. 推理示例 (可选，用于验证)
# ==============================================================================
print("\n--- 步骤 6: 推理示例 (验证微调效果) ---")

# 重新加载基础模型 (需要再次进行4位量化加载)
# 或者直接使用训练好的model对象，但通常会重新加载以模拟部署
from peft import PeftModel

print("正在加载基础模型和微调后的LoRA适配器进行推理...")
base_model_for_inference = AutoModelForCausalLM.from_pretrained(
    config.model_name_or_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
)

# 加载微调后的LoRA适配器
model_for_inference = PeftModel.from_pretrained(base_model_for_inference, final_checkpoint_path)

# 可选：合并LoRA权重到基础模型 (如果希望部署为单一模型文件)
# model_for_inference = model_for_inference.merge_and_unload()

# 确保模型处于评估模式
model_for_inference.eval()

# 测试一个医疗问题
test_messages = [
    {"role": "system", "content": "你是一个乐于助人的医疗助手。"},
    {"role": "user", "content": "糖尿病的症状有哪些？"},
]
input_ids = tokenizer.apply_chat_template(test_messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
input_ids = input_ids.to(model_for_inference.device)

print(f"用户提问: {test_messages[-1]['content']}")

with torch.no_grad():
    outputs = model_for_inference.generate(
        input_ids=input_ids,
        max_new_tokens=256,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
print(f"模型回答:\n{response}")
print("-" * 50)

print("所有阶段已完成。")