import json

# 将单个.json文件中的请求格式转换为数据集
def transform_data(request):
    # 提取特征
    duration = request["http.request.duration"]
    method = request["http.request.method"]
    remoteaddr = request["http.request.remoteaddr"]
    useragent = request["http.request.useragent"]
    uri = request["http.request.uri"]
    # 获取标签
    label = uri.split('/')[1] if len(uri.split('/')) >= 2 else 'default'
    # 返回特征和标签
    return [duration, method, remoteaddr, useragent, label]

def load_json_file(json_file):
    raw_data = []
    with open(json_file, 'r') as f:
        try:
            requests = json.load(f)
            for request in requests:
                data = transform_data(request)
                raw_data.append(data)
        except ValueError as value_err:
            print("ValueError: " + str(value_err) + " in JSON file: " + json_file)
        except json.JSONDecodeError as json_err:
            print("JSONDecodeError: " + str(json_err) + " in JSON file: " + json_file)
    return raw_data
