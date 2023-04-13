import os
import json

# 定义要合并的目录和文件名
directory = "../dataset/dal09-nodes/786b3803"
output_filename = "data.json"

# 创建一个空列表，用于存储所有JSON数据
data = []

# 遍历目录中的所有文件
for filename in os.listdir(directory):
    if filename.endswith(".json"):
        # 打开JSON文件并将其解析为Python对象
        with open(os.path.join(directory, filename)) as f:
            json_data = json.load(f)
        # 将该文件的数据添加到列表中
        data.extend(json_data)

# 将所有数据写入一个新的JSON文件
with open(os.path.join(directory, output_filename), "w") as f:
    json.dump(data, f)
