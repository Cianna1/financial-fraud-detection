from kafka import KafkaProducer

import json
import time

# Kafka 配置
producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

topic = "transactions"  # 保持一致

# 加载历史 Parquet 文件
import pandas as pd

df = pd.read_csv("C:\\Users\\陈一心\\.kaggle\\archive\\creditcard.csv").head(20)
df.to_parquet("historical_transactions.parquet", index=False)


# 模拟发送，每条延迟 0.5 秒
for idx, row in df.iterrows():
    record = row.to_dict()
    producer.send(topic, record)
    print(f"📤 已发送第 {idx+1} 条: {record}")
    time.sleep(0.5)

producer.flush()
print("✅ 所有历史记录已完成发送")

# 等待 Spark Streaming 处理完最后一批（可选延迟 2s）
time.sleep(2)

# 读取 Redis 中处理结果
import redis

r = redis.StrictRedis(host='localhost', port=6379, db=0)
print("\n🧾 Redis 中已处理结果预览（最多显示 10 条）:")
count = 0
for key in r.scan_iter("tx:*"):
    value = r.get(key)
    print(f"🔍 {key.decode()}: {json.loads(value)}")
    count += 1
    if count >= 10:
        break

