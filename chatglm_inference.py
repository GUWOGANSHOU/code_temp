import os
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

# ����Tokenizer
tokenizer = AutoTokenizer.from_pretrained("chatglm-6b", trust_remote_code=True)
# ����ģ��
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

response, history = model.chat(tokenizer, "����#��װ��*��ɫ#����ɫ*ͼ��#����*�㳤#�˷ֿ�", history=[])
print(response)

'''
import json
def generate_summaries(test_set):
    for item in test_set:
        # ��content�ֶ���Ϊ���룬����ģ������ժҪ
        response, history = model.chat(tokenizer, item["content"], history=[])
        # ֱ�ӽ����ɵ�ժҪ�洢��ԭitem��summary�ֶ�
        item["summary"] = response
    return test_set
# ���ú�������ժҪ�����²��Լ�
updated_test_set = generate_summaries(test_set)
# ��ӡ���
for item in updated_test_set:
    print(item)
# ѡ��һ���ļ�����·�������� JSON ����
file_path = 'updated_test_set.json'
# ʹ�� json ģ�齫����д���ļ�
with open(file_path, 'w', encoding='utf-8') as json_file:
    json.dump(updated_test_set, json_file, ensure_ascii=False, indent=4)
print(f"Updated test set has been saved to {file_path}")
'''