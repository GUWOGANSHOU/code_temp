import os
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

# 载入Tokenizer
tokenizer = AutoTokenizer.from_pretrained("chatglm-6b", trust_remote_code=True)
# 载入模型
config = AutoConfig.from_pretrained("chatglm-6b", trust_remote_code=True, pre_seq_len=128)
model = AutoModel.from_pretrained("chatglm-6b", config=config, trust_remote_code=True)
CHECKPOINT_PATH = 'output/adgen-chatglm-6b-pt-128-2e-2/checkpoint-3000'
prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

# Comment out the following line if you don't use quantization
model = model.quantize(4)
model = model.half().cuda()
model.transformer.prefix_encoder.float()
model = model.eval()

response, history = model.chat(tokenizer, "类型#工装裤*颜色#深蓝色*图案#条纹*裤长#八分裤", history=[])
print(response)

'''
import json
def generate_summaries(test_set):
    for item in test_set:
        # 将content字段作为输入，调用模型生成摘要
        response, history = model.chat(tokenizer, item["content"], history=[])
        # 直接将生成的摘要存储在原item的summary字段
        item["summary"] = response
    return test_set
# 调用函数生成摘要并更新测试集
updated_test_set = generate_summaries(test_set)
# 打印结果
for item in updated_test_set:
    print(item)
# 选择一个文件名和路径来保存 JSON 数据
file_path = 'updated_test_set.json'
# 使用 json 模块将数据写入文件
with open(file_path, 'w', encoding='utf-8') as json_file:
    json.dump(updated_test_set, json_file, ensure_ascii=False, indent=4)
print(f"Updated test set has been saved to {file_path}")
'''