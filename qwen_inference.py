from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

# ����Ԥѵ���ķִ���
tokenizer = AutoTokenizer.from_pretrained("Qwen")  # ������Ҫ�滻��ʵ�ʵ�Qwenģ��ID

# ����΢�����ģ��
model = AutoPeftModelForCausalLM.from_pretrained(
    "path_to_adapter",  # ΢��ģ�͵�·��
    device_map="auto",
    trust_remote_code=True
).eval()

# ����
for dialog in test_set:
    conversations = dialog["conversations"]
    for i, conv in enumerate(conversations):
        if conv["from"] == "user":
            # ʹ��ģ�����ɻش�
            response, history = model.chat(tokenizer, conv["value"], history=None)
            # ���ش�洢����һ��assistant�ֶ�
            if i + 1 < len(conversations) and conversations[i + 1]["from"] == "assistant":
                conversations[i + 1]["value"] = response

# ��ӡ���º�Ĳ��Լ�
for dialog in test_set:
    print(dialog)