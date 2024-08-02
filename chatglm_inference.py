# 下载 requirements.txt 安装配置环境
pip install -r requirements.txt
# modelscope 用来下载 chatglm2-6b-int4
pip install modelscope

# 微调还需要下载的文件
pip install rouge_chinese nltk jieba datasets 
# 不一定需要装deepspeed
# pip install "peft<0.8.0" deepspeed

# Ptuning
# bash train.sh

PRE_SEQ_LEN=128
LR=2e-2
NUM_GPUS=1

torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_train \
    --train_file AdvertiseGen/train.json \
    --validation_file AdvertiseGen/dev.json \
    --preprocessing_num_workers 10 \
    --prompt_column content \
    --response_column summary \
    --overwrite_cache \
    --model_name_or_path /root/chatglm2-6b \
    --output_dir output/adgen-chatglm2-6b-pt-$PRE_SEQ_LEN-$LR \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 128 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 1000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4


# bash evaluate.sh

PRE_SEQ_LEN=128
CHECKPOINT=adgen-chatglm2-6b-pt-128-2e-2
STEP=1000
NUM_GPUS=1


torchrun --standalone --nnodes=1 --nproc-per-node=$NUM_GPUS main.py \
    --do_predict \
    --validation_file AdvertiseGen/dev.json \
    --test_file AdvertiseGen/dev.json \
    --overwrite_cache \
    --prompt_column content \
    --response_column summary \
    --model_name_or_path /root/chatglm2-6b \
    --ptuning_checkpoint ./output/$CHECKPOINT/checkpoint-$STEP \
    --output_dir ./output/$CHECKPOINT \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_eval_batch_size 1 \
    --predict_with_generate \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantization_bit 4

import os
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

CHECKPOINT_PATH = "/root/ChatGLM2-6B/ptuning/output/adgen-chatglm2-6b-pt-128-2e-2/checkpoint-1000"
# 载入Tokenizer
tokenizer = AutoTokenizer.from_pretrained("/root/chatglm2-6b", trust_remote_code=True)
config = AutoConfig.from_pretrained("/root/chatglm2-6b", trust_remote_code=True, pre_seq_len=128)
model = AutoModel.from_pretrained("/root/chatglm2-6b", config=config, trust_remote_code=True)
prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

# Comment out the following line if you don't use quantization
model = model.quantize(4)
model = model.cuda()
model = model.eval()

response, history = model.chat(tokenizer, "类型#工装裤*颜色#深蓝色*图案#条纹*裤长#八分裤", history=[])
print(response)

import json
with open('ChatGLM2-6B/ptuning/AdvertiseGen/test.json', 'r', encoding='utf-8') as file:
    test_set = json.load(file)
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
