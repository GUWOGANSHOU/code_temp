#coding=gb18030
import pandas as pd
import json

# 读取CSV文件
csv_file_path = 'datasets/qwen/sample_IM5000-6000.csv'
df = pd.read_csv(csv_file_path, encoding='ANSI')

# 创建一个空列表来保存转换后的数据
data_list = []

# 遍历DataFrame的每一行
for index, row in df.iterrows():
    # 构造新的行数据
    department = row['department'] + '问诊'
    title = row['title']
    ask = row['ask']
    answer = row['answer']

    # 拼接user的value
    user_value = f"{department}，{title}{ask}"

    # 创建对话结构
    conversations = [
        {"from": "user", "value": user_value},
        {"from": "assistant", "value": answer}
    ]

    # 创建完整条目
    entry = {
        "id": f"identity_{index}",
        "conversations": conversations
    }

    # 添加到列表中
    data_list.append(entry)

# 将数据序列化为JSON格式
json_data = json.dumps(data_list, indent=4, ensure_ascii=False)

# 写入JSON文件
json_file_path = './datasets/qwen/sample_IM5000-6000.json'
with open(json_file_path, 'w', encoding='utf-8') as json_file:
    json_file.write(json_data)

print(f"Data has been successfully converted to JSON format and saved to {json_file_path}")