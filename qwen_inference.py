from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

# 加载预训练的分词器
tokenizer = AutoTokenizer.from_pretrained("Qwen")  # 这里需要替换成实际的Qwen模型ID

# 加载微调后的模型
model = AutoPeftModelForCausalLM.from_pretrained(
    "path_to_adapter",  # 微调模型的路径
    device_map="auto",
    trust_remote_code=True
).eval()

# 推理
for dialog in test_set:
    conversations = dialog["conversations"]
    for i, conv in enumerate(conversations):
        if conv["from"] == "user":
            # 使用模型生成回答
            response, history = model.chat(tokenizer, conv["value"], history=None)
            # 将回答存储到下一个assistant字段
            if i + 1 < len(conversations) and conversations[i + 1]["from"] == "assistant":
                conversations[i + 1]["value"] = response

# 打印更新后的测试集
for dialog in test_set:
    print(dialog)