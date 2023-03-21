from sklearn.preprocessing import LabelEncoder
import os
import json

# 将单个.json文件中的请求格式转换为数据集
def transform_data(request):
    # 提取特征
    duration = request["http.request.duration"]
    method = request["http.request.method"]
    remoteaddr = request["http.request.remoteaddr"]
    useragent = request["http.request.useragent"]
    # status = request["http.response.status"]
    # written = request["http.response.written"]
    # 提取标签
    label = request["http.request.uri"]
    # 标签编码
    label_encoder = LabelEncoder()
    label = label_encoder.fit_transform([label])[0]
    # 返回特征和标签
    # return [duration, method, remoteaddr, useragent, status, written, label]
    return [duration, method, remoteaddr, useragent, label]

# 将多个.json文件中的请求格式转换为数据集
# def transform_dataset(json_files):
#     dataset = []
#     try:
#         for json_file in json_files:
#             with open(json_file, 'r') as f:
#                 requests = json.load(f)
#                 for request in requests:
#                     data = transform_data(request)
#                     dataset.append(data)
#     except ValueError as value_err:
#         print("ValueError:"+str(value_err))
#     return dataset

# 将多个.json文件中的请求格式转换为数据集
def transform_dataset(json_files):
    dataset = []
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                requests = json.load(f)
                for request in requests:
                    data = transform_data(request)
                    dataset.append(data)
        except ValueError as value_err:
            print("ValueError: " + str(value_err) + " in JSON file: " + json_file)
        except json.JSONDecodeError as json_err:
            print("JSONDecodeError: " + str(json_err) + " in JSON file: " + json_file)
    return dataset


# 遍历所有文件夹，获取所有.json文件的路径
json_files = []
for root, dirs, files in os.walk("dataset"):
    for file in files:
        if file.endswith(".json"):
            json_file = os.path.join(root, file)
            json_files.append(json_file)

# print(json_files)
# 将所有.json文件中的数据加入到数据集
dataset = transform_dataset(json_files)
# print(dataset)
