# create_random_model.py
import torch
from transformers import LlamaForCausalLM, LlamaConfig
import os

# 读取配置
config = LlamaConfig.from_pretrained("models/llama3_2-1B-scratch")

# 确保配置正确
config.max_position_embeddings = 2048
config.model_max_length = 2048

# 创建随机初始化的模型
model = LlamaForCausalLM(config)

# 保存模型和配置
model.save_pretrained("models/llama3_2-1B-scratch")
config.save_pretrained("models/llama3_2-1B-scratch")

print("Random initialized model created!")
print(f"Model parameters: {model.num_parameters():,}")