#!/usr/bin/env python3
# test_loopllama.py

"""
测试 LoopLlama 模型是否正常工作
"""

import torch
from transformers import AutoTokenizer
from models.loopllama_origin.modeling_llama import LoopLlamaForCausalLM

def test_loopllama_model():
    print("Loading LoopLlama model...")
    
    # 加载模型
    model = LoopLlamaForCausalLM.from_pretrained(
        "models/loopllama_origin",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("models/loopllama_origin")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Model loaded. Loop times: {model.config.loop_times}")
    
    # # 测试推理
    # test_prompt = "Hello, how are you?"
    # inputs = tokenizer(test_prompt, return_tensors="pt")
    # print(f"Input shape: {inputs['input_ids'].shape}")
    # print("Testing inference...")
    # with torch.no_grad():
    #     outputs = model.generate(
    #         **inputs,
    #         max_new_tokens=50,
    #         do_sample=True,
    #         temperature=0.7,
    #         use_cache=True
    #     )
    
    # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(f"Input: {test_prompt}")
    # print(f"Output: {response}")
    
    # 测试训练模式
    print("Testing training mode...")
    model.train()
    
    # 创建批次数据
    batch_inputs = tokenizer(
        ["Hello world", "How are you?", "Test sentence"], 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=128
    )
    
    # 前向传播
    with torch.cuda.amp.autocast():
        outputs = model(**batch_inputs, labels=batch_inputs["input_ids"])
        loss = outputs.loss
    
    print(f"Training loss: {loss.item():.4f}")
    
    # 测试反向传播
    loss.backward()
    print("Backward pass successful!")
    
    print("All tests passed!")
    return True

if __name__ == "__main__":
    test_loopllama_model()