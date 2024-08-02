# pip install mpi4py -i http://pypi.doubanio.com/simple/ --trusted-host pypi.doubanio.com
# # ERROR: Could not build wheels for mpi4py, which is required to install pyproject.toml-based projects，出现此错误的话，conda安装
# sudo apt update
# sudo apt-get install libopenmpi-dev
# # 执行以上两条命令后， 重新下载安装mpi4py命令，发现还是安装失败，改用conda的方式安装
# conda install mpi4py

# pip install optimum
# pip install auto-gptq
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple auto-gptq
# pip install tiktoken


#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('Qwen/Qwen-7B-Chat-Int4', local_dir="/root/Qwen-7B-Chat-Int4")


# 微调训练
# MODEL="/root/Qwen-7B-Chat-Int4" 
# DATA="/root/datasets/qwen/val_sample_IM5000-6000.json"

# bash finetune/finetune_lora_single_gpu.sh


# 推理
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
import json

# 加载预训练的分词器
tokenizer = AutoTokenizer.from_pretrained("/root/Qwen-7B-Chat-Int4")  # 这里需要替换成实际的Qwen模型ID

# 加载微调后的模型
model = AutoPeftModelForCausalLM.from_pretrained(
    "output_qwen/test1",  # 微调模型的路径
    device_map="auto",
    trust_remote_code=True
).eval()

# 假设你的JSON数据存储在一个名为data.json的文件中
with open('datasets/qwen/test.json', 'r', encoding='utf-8') as file:
    test_set = json.load(file)

# 引入你的模型和tokenizer
# model 和 tokenizer 应该是你预先训练好的模型和相应的分词器
# 这里我们假设它们已经被定义并且可用
# from your_model_module import model, tokenizer

for dialog in test_set:
    conversations = dialog["conversations"]
    history = None
    for i, conv in enumerate(conversations):
        if conv["from"] == "user":
            # 使用模型生成回答
            response, history = model.chat(tokenizer, conv["value"], history=history)
            # 将回答存储到下一个assistant字段
            if i + 1 < len(conversations) and conversations[i + 1]["from"] == "assistant":
                conversations[i + 1]["value"] = response

# 打印更新后的测试集
for dialog in test_set:
    print(dialog)
# 将更新后的数据集保存到新的JSON文件中
with open('updated_data.json', 'w', encoding='utf-8') as file:
    json.dump(test_set, file, ensure_ascii=False, indent=4)

print("数据已成功更新并保存到 updated_data.json 文件。")
