from collections import Counter
import json
import os
from tqdm import tqdm

data = []
# 加载数据集并解析为JSON格式。
json_files = []
for root, dirs, files in os.walk("../dataset/node-9e"):
    for file in files:
        if file.endswith(".json"):
            json_file = os.path.join(root, file)
            json_files.append(json_file)

for json_file in tqdm(json_files):
    with open(json_file, "r") as f:
        json_object = json.load(f)
        data.extend(json_object)

uris = []
for item in data:
    uri_parts = item["http.request.uri"].split("/")[2:3] or ["default"]
    uris.append("/".join(uri_parts))

uri_counts = Counter(uris)
total_count = sum(uri_counts.values())
print(f"共有{total_count}种不同的v2后面两个部分，记录如下：")
print(f"共有{len(uri_counts)}种不同的v2后面两个部分，记录如下：")
for uri, count in uri_counts.most_common():
    print(f"{uri}: {count} 次，占比{count/total_count:.2%}")
