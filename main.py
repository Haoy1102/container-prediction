import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
print(sys.path)

from preprocessing.data_transform import load_json_file
from preprocessing.data_preprocess import preprocess_data
from train import train_model
from test_model import test_model
from utils.logger import get_logger
import tqdm
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


logger = get_logger(__name__)

if __name__ == '__main__':
    json_files = []
    for root, dirs, files in os.walk("./dataset"):
        for file in files:
            if file.endswith(".json"):
                json_file = os.path.join(root, file)
                json_files.append(json_file)

    raw_data = []
    for json_file in tqdm.tqdm(json_files, total=len(json_files)):
        raw_data += load_json_file(json_file)

    features, labels = preprocess_data(raw_data)

    # 划分训练集和测试集
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42,shuffle=True)

    logger.info("数据预处理完成")

    # 训练模型
    model = train_model(X_train, X_test, y_train, y_test)

    # 在测试集上进行评估
    test_acc = test_model(X_test, y_test, model)

    logger.info("测试集准确率为：{:.2f}%".format(test_acc * 100))
