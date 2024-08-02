#coding=gb18030
import pandas as pd
import json

# ��ȡCSV�ļ�
csv_file_path = 'datasets/qwen/sample_IM5000-6000.csv'
df = pd.read_csv(csv_file_path, encoding='ANSI')

# ����һ�����б�������ת���������
data_list = []

# ����DataFrame��ÿһ��
for index, row in df.iterrows():
    # �����µ�������
    department = row['department'] + '����'
    title = row['title']
    ask = row['ask']
    answer = row['answer']

    # ƴ��user��value
    user_value = f"{department}��{title}{ask}"

    # �����Ի��ṹ
    conversations = [
        {"from": "user", "value": user_value},
        {"from": "assistant", "value": answer}
    ]

    # ����������Ŀ
    entry = {
        "id": f"identity_{index}",
        "conversations": conversations
    }

    # ��ӵ��б���
    data_list.append(entry)

# ���������л�ΪJSON��ʽ
json_data = json.dumps(data_list, indent=4, ensure_ascii=False)

# д��JSON�ļ�
json_file_path = './datasets/qwen/sample_IM5000-6000.json'
with open(json_file_path, 'w', encoding='utf-8') as json_file:
    json_file.write(json_data)

print(f"Data has been successfully converted to JSON format and saved to {json_file_path}")