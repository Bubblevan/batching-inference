import pyarrow.parquet as pq
import os
import numpy as np

DATASET_DIR = "/root/autodl-pub/datasets/lmsys-chat-1m/data"

def test_load_dataset(dataset_dir):
    files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".parquet")]
    for file in files:
        print(f"Testing file: {file}")
        table = pq.read_table(file)
        df = table.to_pandas()

        # 打印字段名
        print("Available columns:", df.columns)

        # 遍历每行数据
        for _, row in df.iterrows():
            conversation = row.get("conversation")
            if conversation is None:
                print("Conversation field is None.")
                continue
            elif isinstance(conversation, list):
                print("Conversation field is a list.")
            elif isinstance(conversation, np.ndarray):
                print("Conversation field is a numpy.ndarray.")
                conversation = conversation.tolist()  # 转换为 Python 列表
            else:
                print("Conversation field is of unexpected type:", type(conversation))
                continue

            # 确认是否为有效对话
            if isinstance(conversation, list) and len(conversation) > 0:
                print("Valid conversation detected.")
                for turn in conversation:
                    print("Turn structure:", turn)
                    print("Role:", turn.get("role"))
                    print("Content:", turn.get("content"))
                break  # 打印一个完整对话后退出
            else:
                print("Invalid conversation field:", conversation)
        break  # 测试第一个文件后退出

test_load_dataset(DATASET_DIR)
