import json
import os

original_folder_path = "./dev-mon01"
new_folder_path = "./dev-nodes"
for filename in os.listdir(original_folder_path):
    if filename.endswith(".json"):
        with open(os.path.join(original_folder_path, filename), 'r') as f:
            data = json.load(f)

        if not data:
            continue

        for obj in data:
            host_value = obj["host"]
            new_host_folder_path = os.path.join(new_folder_path, host_value)

            if not os.path.exists(new_host_folder_path):
                os.makedirs(new_host_folder_path)
            # 修改写入方式为追加模式
            with open(os.path.join(new_host_folder_path, filename), 'a') as f:
                # 将当前JSON对象转换成字符串并写入文件中
                json.dump(obj, f)

                # 在每个JSON对象之间添加逗号分隔符
                f.write(",")

# 遍历所有主机名下的新分类.json文件，删除最后一个逗号，并在开头和结尾添加方括号以保持原有JSON格式
for root, dirs, files in os.walk(new_folder_path):
    for file_name in files:
        if file_name.endswith(".json"):
            file_fullpath = os.path.join(root, file_name)

            # 读取整个文件内容，并去掉末尾最后一个逗号
            with open(file_fullpath, 'rb+') as fp:
                fp.seek(-1, os.SEEK_END)
                fp.truncate()

            # 在开头和结尾添加方括号以保持原有JSON格式
            with open(file_fullpath, 'r+') as fp:
                fp.seek(0, 0)
                fp.write("[")
                fp.write(content)
                fp.seek(0, os.SEEK_END)
                fp.write("]")