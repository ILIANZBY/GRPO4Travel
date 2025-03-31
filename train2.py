import os
from trl import GRPOConfig, GRPOTrainer
from reward import reward_function
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from glob import glob

# 强制使用GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 检查GPU可用性
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.current_device()}")  # 应该显示0

# 8-bit量化配置
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=["lm_head"],
)

# 查找最新检查点
# def find_latest_checkpoint(output_dir):
#     checkpoint_dirs = glob(os.path.join(output_dir, "checkpoint-*"))
#     if not checkpoint_dirs:
#         return None
#     return max(checkpoint_dirs, key=lambda x: int(x.split("-")[-1]))

# latest_checkpoint = find_latest_checkpoint("./output")
# if latest_checkpoint:
#     print(f"Resuming from checkpoint: {latest_checkpoint}")

# 训练参数配置 (batch_size增加为两倍)
training_args = GRPOConfig(
    output_dir="/share/home/wuqingyao_zhangboyang/grpo/n_output",
    per_device_train_batch_size=8,      # 从4增加到8 (两倍)
    gradient_accumulation_steps=1,      # 如果保持总batch size不变，可以减半梯度累积步数
    num_generations=4,
    bf16=True,
    gradient_checkpointing=True,
    optim="adamw_torch_fused",
    max_prompt_length=4096,
    max_completion_length=512,
    learning_rate=3e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    num_train_epochs=3,
    logging_steps=50,
    save_steps=500,
    # resume_from_checkpoint=latest_checkpoint if latest_checkpoint else None
    
)

# 加载数据集
dataset = load_dataset("json", 
                      data_files="/share/home/wuqingyao_zhangboyang/grpo/n_dataset/n_grpo.json",
                      keep_in_memory=True,
                      num_proc=8)["train"]
dataset = dataset.train_test_split(test_size=0.01)

# 加载模型并强制分配到GPU 0
model = AutoModelForCausalLM.from_pretrained(
    "/share/home/wuqingyao_zhangboyang/2745",
    quantization_config=quant_config,
    device_map={"": 0},  # 强制所有内容分配到device 0
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
)

peft_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "o_proj"],  # 移除了重复的"v_proj"
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

# 初始化训练器
trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    reward_funcs=reward_function,
)

# 开始训练
trainer.train()