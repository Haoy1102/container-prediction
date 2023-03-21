from sklearn.preprocessing import LabelEncoder



# 将请求格式转换为数据集
def transform_data(request):
    # 提取特征
    duration = request["http.request.duration"]
    method = request["http.request.method"]
    remoteaddr = request["http.request.remoteaddr"]
    useragent = request["http.request.useragent"]
    status = request["http.response.status"]
    written = request["http.response.written"]
    # 提取标签
    label = request["http.request.uri"]
    # 标签编码
    label_encoder = LabelEncoder()
    label = label_encoder.fit_transform([label])[0]
    # 返回特征和标签
    return [duration, method, remoteaddr, useragent, status, written, label]

# 将多个请求格式转换为数据集
def transform_dataset(requests):
    dataset = []
    for request in requests:
        data = transform_data(request)
        dataset.append(data)
    return dataset

# 示例数据
requests = [
    {"host": "033292e5",
     "http.request.duration": 1.40929383,
     "http.request.method": "PUT",
     "http.request.remoteaddr": "e1852826",
     "http.request.uri": "v2/503550a5/d1002b86/manifests/817c8a39",
     "http.request.useragent": "docker/17.04.0-ce go/go1.7.5 git-commit/4845c56 kernel/4.4.0-83-generic os/linux arch/amd64 UpstreamClient(Docker-Client/17.04.0-ce (linux))",
     "http.response.status": 201,
     "http.response.written": 524,
     "id": "baeef7c1eb",
     "timestamp": "2017-09-05T17:23:42.900Z"
    }
]

# 转换为数据集
dataset = transform_dataset(requests)
print(dataset)
