import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import re # 导入正则表达式库

# ==============================================================================
# 0. 配置参数
# ==============================================================================
class Config:
    base_model_path = "/hy-tmp/Qwen-7B-Chat"
    lora_adapter_path = "./qwen_7b_medical_lora_output/final_checkpoint"
    bf16 = True
    fp16 = False

config = Config()

print("--- Qlora微调模型推理脚本 ---")
print(f"基础模型路径: {config.base_model_path}")
print(f"LoRA适配器路径: {config.lora_adapter_path}")
print(f"使用 BF16 计算: {config.bf16}, 使用 FP16 计算: {config.fp16}")
print("-" * 50)

# ==============================================================================
# 1. 配置4位量化 (与训练时一致)
# ==============================================================================
print("--- 步骤 1: 配置4位量化 ---")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
)
print("BitsAndBytesConfig 已设置。")
print("-" * 50)

# ==============================================================================
# 2. 加载分词器 (与训练时一致)
# ==============================================================================
print("--- 步骤 2: 正在加载分词器并配置 ---")
tokenizer = AutoTokenizer.from_pretrained(
    config.base_model_path,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = '<|endoftext|>'
    tokenizer.pad_token_id = 151643
if tokenizer.eos_token is None:
    tokenizer.eos_token = '<|endoftext|>'
    tokenizer.eos_token_id = 151643
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
print("分词器已加载并配置。")
print("-" * 50)

# ==============================================================================
# 3. 加载基础模型和微调后的LoRA适配器
# ==============================================================================
print("--- 步骤 3: 正在加载基础模型 (4位量化) 和微调后的LoRA适配器 ---")

base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 if config.bf16 else torch.float16,
)
print("基础模型已加载。")

model = PeftModel.from_pretrained(base_model, config.lora_adapter_path)
print(f"LoRA适配器已从 {config.lora_adapter_path} 加载并注入基础模型。")

model.eval()
model.config.use_cache = True
print("模型已设置为评估模式，并启用缓存。")
print("-" * 50)

# ==============================================================================
# 4. 进行推理
# ==============================================================================
print("--- 步骤 4: 进行推理 ---")

# --- 关键修改：调整 system_prompt，明确要求中文回答 ---
def generate_response(user_question, system_prompt="你是一个乐于助人的医疗助手，请用中文回答。"):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
    ]
    
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    input_ids = input_ids.to(model.device)

    attention_mask = torch.ones_like(input_ids).to(model.device)
    
    print(f"\n用户提问: {user_question}")

    stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|im_end|>')]
    stop_ids = [id for id in stop_ids if id is not None]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1024, # --- 关键修改：增加 max_new_tokens 到 1024 ---
            do_sample=True,
            top_p=0.8,
            temperature=0.7,
            num_return_sequences=1,
            eos_token_id=stop_ids,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.2,
        )

    response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=False)
    
    # --- 关键修改：更彻底的后处理 ---
    response = response.replace('<|im_start|>', '').replace('<|im_end|>', '').strip()
    response = response.replace('<|endoftext|>', '').strip()
    response = re.sub(r'assistant', '', response).strip() # 移除可能残留的 "assistant"

    # 移除所有看起来像URL路径或模板标签的结构
    # 匹配以冒号开头，后面跟着斜杠和任意非空白字符的模式
    response = re.sub(r':\s*/\S+', '', response).strip()
    response = re.sub(r'\[/\S+\]', '', response).strip()
    response = re.sub(r'\[\d+\]', '', response).strip()

    # 处理重复内容 (保持现有逻辑)
    sentences = re.split(r'(?<=[。？！])\s*', response)
    unique_sentences = []
    seen_phrases = set()
    for sentence in sentences:
        if sentence and sentence not in seen_phrases:
            unique_sentences.append(sentence)
            seen_phrases.add(sentence)
        elif sentence and sentence in seen_phrases:
            if sentence != sentences[-1]:
                break
    response = ' '.join(unique_sentences).strip()
    
    print(f"模型回答:\n{response}")
    print("-" * 50)
    return response

# --- 测试示例 ---
generate_response("糖尿病的症状有哪些？")
generate_response("什么是高血压？如何预防？")
generate_response("请解释一下癌症的早期症状。")

print("推理脚本运行完成。")