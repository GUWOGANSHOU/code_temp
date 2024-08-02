# coding=gb18030
import pandas as pd
import json
from sklearn.model_selection import train_test_split

# 读取CSV文件
csv_file_path = 'datasets/qwen/sample_IM5000-6000.csv'
df = pd.read_csv(csv_file_path, encoding='ANSI')

# 划分数据集
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# 转换训练集数据
train_data_list = []
for index, row in train_df.iterrows():
    department = row['department'] + '问诊'
    title = row['title']
    ask = row['ask']
    answer = row['answer']

    user_value = f"{department}，{title}{ask}"

    conversations = [
        {"from": "user", "value": user_value},
        {"from": "assistant", "value": answer}
    ]

    entry = {
        "id": f"identity_{index}",
        "conversations": conversations
    }

    train_data_list.append(entry)

# 将训练集数据序列化为JSON格式
train_json_data = json.dumps(train_data_list, indent=4, ensure_ascii=False)

# 写入训练集JSON文件
train_json_file_path = './datasets/qwen/train_sample_IM5000-6000.json'
with open(train_json_file_path, 'w', encoding='utf-8') as json_file:
    json_file.write(train_json_data)

# 转换验证集数据
val_data_list = []
for index, row in val_df.iterrows():
    department = row['department'] + '问诊'
    title = row['title']
    ask = row['ask']
    answer = row['answer']

    user_value = f"{department}，{title}{ask}"

    conversations = [
        {"from": "user", "value": user_value},
        {"from": "assistant", "value": answer}
    ]

    entry = {
        "id": f"identity_{index}",
        "conversations": conversations
    }

    val_data_list.append(entry)

# 将验证集数据序列化为JSON格式
val_json_data = json.dumps(val_data_list, indent=4, ensure_ascii=False)

# 写入验证集JSON文件
val_json_file_path = './datasets/qwen/val_sample_IM5000-6000.json'
with open(val_json_file_path, 'w', encoding='utf-8') as json_file:
    json_file.write(val_json_data)

print(f"Train data has been successfully converted to JSON format and saved to {train_json_file_path}")
print(f"Validation data has been successfully converted to JSON format and saved to {val_json_file_path}")
