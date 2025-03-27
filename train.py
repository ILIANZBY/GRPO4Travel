from trl import GRPOConfig,GRPOTrainer
from reward import reward_function


training_args = GRPOConfig(
    # 基础路径配置
    output_dir="/share/home/wuqingyao_zhangboyang/grpo/output",
    logging_dir="./logs",  # 新增独立日志目录
    report_to=["tensorboard"],  # 增加可视化监控
    
    # 批处理与内存优化
    per_device_train_batch_size=8,  # 增大batch size (原4→8)
    gradient_accumulation_steps=2,  # 新增梯度累积
    fp16=True,  # 启用混合精度训练
    
    # 学习率调度
    learning_rate=3e-5,  # 适当提高学习率 (原1e-5→3e-5)
    lr_scheduler_type="cosine_with_restarts",  # 新增余弦退火
    warmup_ratio=0.1,  # 新增学习率预热
    
    # 强化学习参数优化
    beta=0.2,  # 调整KL散度系数 (原0.1→0.2)
    num_generations=8,  # 增加生成样本数 (原8→12)
    # generation_kwargs={
    #     "top_p": 0.9,  # 新增核采样
    #     "temperature": 0.7,  # 控制多样性
    #     "repetition_penalty": 1.2  # 防止重复
    # },
    
    # 序列长度配置
    max_prompt_length=512,
    max_completion_length=384,  # 适当增加 (原256→384)
    # truncation_mode="keep_end",  # 重要信息通常在末尾
    
    # 训练过程控制
    num_train_epochs=5,  # 明确训练轮次
    save_steps=500,  # 保存间隔
    logging_steps=50,  # 减少日志频率 (原10→50)
    evaluation_strategy="steps",  # 新增评估策略
    eval_steps=200,  # 每200步评估
    
    # 硬件优化
    gradient_checkpointing=True,  # 内存优化
    optim="adamw_torch_fused",  # 使用融合优化器
    torch_compile=True  # 启用模型编译
)
from datasets import load_dataset

# 加载 JSON 文件（自动识别格式）
dataset = load_dataset("json", data_files="/share/home/wuqingyao_zhangboyang/grpo/dataset/3000_grpo.json")["train"]
split_datasets = dataset.train_test_split(test_size=0.01, seed=42)
train_dataset = split_datasets["train"]
test_dataset = split_datasets["test"]
trainer = GRPOTrainer(
    model="/share/home/wuqingyao_zhangboyang/.xinference/cache/qwen2_5-pytorch-7b",
    reward_funcs=reward_function,  
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)
trainer.train()