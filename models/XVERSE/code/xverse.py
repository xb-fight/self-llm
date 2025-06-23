import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig


# 加载预训练的分词器和模型
model_path = "xverse/XVERSE-7B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).cuda()
model.generation_config = GenerationConfig.from_pretrained(model_path)

# 使用 INT8、INT4 进行量化推理 
# model = model.quantize(8).cuda()
model = model.quantize(4).cuda()

model = model.eval()

print("=============Welcome to XVERSE chatbot, type 'exit' to exit.=============")


# 设置多轮对话
history = []
while True:
    user_input = input("\n帅哥美女请输入: ")
    if user_input.lower() == "exit":
        break

    # 记录用户输入
    history.append({"role": "user", "content": user_input})

    # 获取模型回复
    response = model.chat(tokenizer, history)
    print("\nXVERSE-7B-Chat: {}".format(response))

    # 保存模型回复，形成多轮历史
    history.append({"role": "assistant", "content": response})

