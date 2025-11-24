本手册旨在指导用户在有限的 GPU 显存（例如 RTX 3080 10GB）上，
使用 QLoRA 技术高效微调通义千问-7B-Chat (Qwen-7B-Chat) 大型语言模型。
QLoRA 通过 4 位量化加载基础模型并结合 LoRA 适配器训练，极大地降低了显存需求，使得在消费级 GPU 上微调大型模型成为可能
使用的环境：python 3.11
cuda 12.1 
pytorch 2.4.0+cu121
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install transformers==4.51.3 peft==0.15.0 accelerate==1.12.0 bitsandbytes==0.48.2 datasets==4.4.1 pandas pyarrow tiktoken einops transformers_stream_generator==0.0.4 modelscope

下载基础模型
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Qwen/Qwen-7B-Chat', local_dir='/hy-tmp/Qwen-7B-Chat', local_dir_use_symlinks=False)"

下载数据集
modelscope download --dataset Shekswess/llama2_medical_meadow_wikidoc_instruct_dataset --local_dir /hy-tmp/llama2_medical_meadow_wikidoc_instruct_dataset_files

preprocess_data.py 数据预处理 （将原始数据集格式化为 Qwen 聊天模板，并进行分词）
load_qlora_model.py 模型加载与LoRA配置（以 4 位量化加载 Qwen-7B-Chat 模型，并注入 LoRA 适配器）
train_qlora_full.py 训练过程 （使用 transformers.Trainer 启动微调训练）
inference_qlora.py 推理与部署准备（使用微调后的 LoRA 适配器进行推理，并准备打包）


