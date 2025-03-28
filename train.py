from trl import GRPOConfig, GRPOTrainer
from reward import reward_function
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
# 检查GPU可用性
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

# 8-bit量化配置 (比4-bit更稳定，适合大多数情况)
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,          # 启用8-bit量化
    llm_int8_threshold=6.0,     # 数值稳定性阈值
    llm_int8_skip_modules=["lm_head"],  # 跳过输出层量化（避免生成质量下降）
)



# 训练参数配置 (优化显存)
training_args = GRPOConfig(
    output_dir="./output",
    per_device_train_batch_size=4,      # 单卡batch_size
    gradient_accumulation_steps=2,      # 梯度累积
    num_generations=4,                  # 每提示生成2个样本
    bf16=True, 
    # 显存优化关键参数
    gradient_checkpointing=True,        # 梯度检查点
    optim="adamw_torch_fused",          # 融合优化器
    
    # 序列长度
    max_prompt_length=4096,             # 输入最大长度
    max_completion_length=512,          # 输出最大长度
    
    # 其他配置
    learning_rate=3e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    num_train_epochs=3,
    logging_steps=50,
    save_steps=500,
)

# 加载数据集
# 使用内存映射和预加载
dataset = load_dataset("json", data_files="/share/home/wuqingyao_zhangboyang/grpo/dataset/3000_grpo.json",keep_in_memory=True,num_proc=8)["train"]
dataset = dataset.train_test_split(test_size=0.01)

# 加载8-bit量化模型
# 使用flash_atten2
model = AutoModelForCausalLM.from_pretrained(
    "/share/home/wuqingyao_zhangboyang/2745",
    quantization_config=quant_config,    # 应用8-bit量化
    device_map="auto",                  # 自动分配设备
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2"
)
peft_config = LoraConfig(
    r=8,                  # 秩
    lora_alpha=32,        # 缩放系数
    target_modules=["q_proj", "v_proj", "v_proj", "o_proj"],  # 目标层（根据模型结构调整）
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