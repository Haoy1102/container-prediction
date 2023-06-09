from collections import Counter
import json
import os
from tqdm import tqdm


data = []
# 加载数据集并解析为JSON格式。
json_files = []
for root, dirs, files in os.walk("../dataset/node-dal09-78-6.21"):
    for file in files:
        if file.endswith(".json"):
            json_file = os.path.join(root, file)
            json_files.append(json_file)

for json_file in tqdm(json_files):
    with open(json_file, "r") as f:
        json_object = json.load(f)
        data.extend(json_object)

import datetime

start_time = datetime.datetime(2017, 6, 21, 6, 0, 0)
end_time = datetime.datetime(2017, 6, 21, 12, 0, 0)

filtered_data = []
for item in data:
    timestamp_str = item["timestamp"]
    timestamp = datetime.datetime.fromisoformat(timestamp_str[:-1])
    if start_time <= timestamp <= end_time:
        filtered_data.append(item)



uris = []
for item in filtered_data:
    uri_parts = item["http.request.uri"].split("/")[1:2] or ["default"]
    uris.append("/".join(uri_parts))

uri_counts = Counter(uris)
total_count = sum(uri_counts.values())

# 设置阈值
threshold = 270
other_count = 0
other_uri = "else"

# 创建一个新的字典来存储合并后的uri_counts
new_uri_counts = {}

# 统计出现次数低于阈值的uri
for uri, count in uri_counts.items():
    if count < threshold:
        other_count += count
    else:
        new_uri_counts[uri] = count

# 将出现次数低于阈值的uri合并为else
new_uri_counts[other_uri] = other_count

# 将new_uri_counts字典转换为Counter对象
new_uri_counts = Counter(new_uri_counts)

print(f"共有{total_count}条数据")
print(f"共有{len(new_uri_counts)}种不同的v2后面两个部分，记录如下：")
for uri, count in new_uri_counts.most_common():
    print(f"{uri}: {count} 次，占比{count/total_count:.2%}")

import json

# 创建一个列表来存储结果
result = []

# 遍历筛选后的数据，将uri和时间戳添加到result列表中
for item in filtered_data:
    timestamp_str = item["timestamp"]
    timestamp = datetime.datetime.fromisoformat(timestamp_str[:-1])
    if start_time <= timestamp <= end_time:
        uri_parts = item["http.request.uri"].split("/")[1:2] or ["default"]
        uri = "/".join(uri_parts)
        result.append({"uri": uri, "timestamp": timestamp_str})

# 将结果保存为JSON文件
with open("filtered_data.json", "w") as f:
    json.dump(result, f)


