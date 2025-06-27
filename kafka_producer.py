# kafka_producer.py
from kafka import KafkaProducer
import pandas as pd
import json
import time

# 1. Kafka 配置
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)
topic = "transactions"

# 2. 加载数据（信用卡欺诈数据集）
df = pd.read_csv("C:\\Users\\陈一心\\.kaggle\\archive\\creditcard.csv")

# 3. 提取特征列（不含 Time 和 Class）
features = [col for col in df.columns if col.startswith("V")] + ["Amount"]
sample_df = df.sample(n=50, random_state=42)  # 可调整条数

# 4. 模拟发送数据
for _, row in sample_df.iterrows():
    msg = {col: float(row[col]) for col in features}
    producer.send(topic, msg)
    print(f"[Kafka] 已发送: {msg}")
    time.sleep(1)  # 每秒一条，模拟实时流

producer.flush()
producer.close()
print("✅ Kafka Producer 发送完成")
