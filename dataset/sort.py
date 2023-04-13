import json
import os

# 指定文件夹路径和新文件名
folder_path = "./node-dal09-78-6.21"
new_file_name = "test-dal09-78-6.21/filtered_data.json"

# 循环读取所有json文件
data = []
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r") as f:
            data += json.load(f)

# 按照时间戳排序
data.sort(key=lambda x: x["timestamp"])

# 写入新文件，使用indent参数实现缩进格式
with open(new_file_name, "w") as f:
    json.dump(data, f, indent=4)

