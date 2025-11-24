import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model # 移除 prepare_model_for_kbit_training
import bitsandbytes as bnb # 导入bnb以检查4bit线性层

# ==============================================================================
# 0. 配置参数
# ==============================================================================
class Config:
    # 模型配置 - 使用你本地完整的Qwen模型路径
    model_name_or_path = "/hy-tmp/Qwen-7B-Chat" # 请确保这是你完整下载的模型路径
    
    # LoRA配置
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    # Qwen模型默认的线性层名称，用于LoRA注入
    # 这里先保留，但我们会在后面动态检查并更新
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    # 量化和数据类型配置
    bf16 = True # 启用bfloat16计算，如果GPU支持 (RTX 30系列及以上通常支持)
    fp16 = False # 如果bf16=False，则设置为True

# 实例化配置
config = Config()

print("--- 阶段二: 加载量化模型并配置LoRA适配器 ---")
print(f"模型路径: {config.model_name_or_path}")
print(f"LoRA Rank (r): {config.lora_r}, LoRA Alpha: {config.lora_alpha}")
print(f"使用 BF16 计算: {config.bf16}, 使用 FP16 计算: {config.fp16}")
print("-" * 50)

# ==============================================================================
# 1. 配置4位量化
# ==============================================================================
print("--- 步骤 1: 配置4位量化 ---")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", # NF4量化，Qlora论文推荐
    bnb_4bit_use_double_quant=True, # 双量化，进一步节省显存
    bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16, # 计算数据类型
)
print("BitsAndBytesConfig 已设置:")
print(bnb_config)
print("-" * 50)

# ==============================================================================
# 2. 加载基础模型 (4位量化)
# ==============================================================================
print(f"--- 步骤 2: 正在加载基础模型 ({config.model_name_or_path}) 并进行4位量化 ---")
model = AutoModelForCausalLM.from_pretrained(
    config.model_name_or_path,
    quantization_config=bnb_config,
    device_map="auto", # 自动分配到可用GPU
    trust_remote_code=True, # Qwen模型需要此参数
    torch_dtype=torch.bfloat16 if config.bf16 else torch.float16, # 模型加载时的数据类型
)
print("基础模型已加载并量化。")
print(f"模型类型: {type(model)}")
print(f"模型设备: {model.device}")
print("-" * 50)

# ==============================================================================
# 3. 准备模型进行k-bit训练 (手动处理，避免嵌入层OOM)
# ==============================================================================
print("--- 步骤 3: 准备模型进行k-bit训练 (手动处理，避免嵌入层OOM) ---")

# 1. 启用梯度检查点，大幅节省显存，但会略微降低训练速度
model.gradient_checkpointing_enable()
print("梯度检查点已启用。")

# 2. 启用输入嵌入层的梯度计算 (对于LoRA微调量化模型是必需的)
model.enable_input_require_grads()
print("输入嵌入层的梯度计算已启用。")

print("模型已为k-bit训练准备就绪 (手动配置)。")
print("-" * 50)

# --- 关键修改：检查模型中的线性层名称 ---
print("--- 调试步骤: 检查模型中的线性层名称 ---")
# 收集所有线性层的名称
linear_layers = []
for name, module in model.named_modules():
    # 检查是否是常规的torch.nn.Linear层或bitsandbytes的4bit线性层
    if isinstance(module, (torch.nn.Linear, bnb.nn.Linear4bit)):
        linear_layers.append(name)

print("模型中发现的线性层名称:")
for layer_name in linear_layers:
    print(f"- {layer_name}")

# 根据Qwen的常见结构，筛选出我们通常希望进行LoRA微调的层
# Qwen通常是注意力层和MLP层中的线性投影
# 常见的模式是 c_attn, c_proj, w1, w2, w3
# 或者像Llama一样是 q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
# 我们需要从上面打印的列表中找到最匹配的
# 举例，如果Qwen的线性层是 transformer.h.0.attn.c_attn，那么我们需要的target_module是 c_attn
# 这里我们尝试自动提取最后一个点号后的名称
extracted_target_modules = set()
for name in linear_layers:
    last_part = name.split('.')[-1]
    # 过滤掉一些不希望LoRA的层，例如 lm_head
    if last_part not in ["lm_head", "embed_tokens"]:
        extracted_target_modules.add(last_part)

# 将提取到的模块名称转换为列表
config.lora_target_modules = list(extracted_target_modules)
print(f"\n根据模型结构自动提取的LoRA目标模块: {config.lora_target_modules}")
print("请确认这些模块名称是否符合预期。")
print("-" * 50)
# ----------------------------------------

# ==============================================================================
# 4. 定义LoRA配置并注入模型
# ==============================================================================
print("--- 步骤 4: 定义LoRA配置并注入模型 ---")
lora_config = LoraConfig(
    r=config.lora_r,
    lora_alpha=config.lora_alpha,
    lora_dropout=config.lora_dropout,
    bias="none", # 不对bias进行LoRA
    task_type="CAUSAL_LM", # 任务类型是因果语言模型
    target_modules=config.lora_target_modules, # 使用动态确定的LoRA注入层
)
print("LoRAConfig 已设置:")
print(lora_config)

# 将LoRA适配器添加到模型
model = get_peft_model(model, lora_config)
print("LoRA适配器已成功注入模型。")
print(f"模型类型现在是: {type(model)}") # 应该变为 PeftModelForCausalLM
print("-" * 50)

# ==============================================================================
# 5. 验证LoRA设置 (打印可训练参数)
# ==============================================================================
print("--- 步骤 5: 验证LoRA设置 (打印可训练参数) ---")
# 打印模型的可训练参数，检查是否只有LoRA参数被训练
model.print_trainable_parameters()
print("你应该看到可训练参数的数量远小于总参数数量，这表明只有LoRA适配器是可训练的。")
print("-" * 50)

print("阶段二完成：量化模型已加载，LoRA适配器已配置并注入。")

# 可选：加载分词器，因为在推理时也需要
print("--- 额外步骤: 加载分词器 ---")
tokenizer = AutoTokenizer.from_pretrained(
    config.model_name_or_path,
    trust_remote_code=True
)

# 手动设置Qwen的pad_token和eos_token，以及chat_template
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

# 打印模型在显存中的占用 (估算)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"模型总参数数量 (包括量化部分): {total_params / 1e9:.2f} B")
print(f"可训练参数数量 (LoRA): {trainable_params / 1e6:.2f} M")
estimated_vram_model = (total_params / 1e9) * 0.5
estimated_vram_lora = (trainable_params / 1e9) * 2
print(f"模型权重估算显存占用: {estimated_vram_model:.2f} GB (基础模型) + {estimated_vram_lora:.2f} GB (LoRA适配器)")
print(f"总模型权重估算显存占用: {estimated_vram_model + estimated_vram_lora:.2f} GB")