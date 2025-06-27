# streaming.py
# Kafka + Spark Structured Streaming + XGBoost + Redis + durable_rules

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, DoubleType
import redis
import json
from durable.lang import assert_fact  # 替换 post
import joblib
import numpy as np
import traceback
import os

# 环境变量配置（如需）
os.environ["HADOOP_HOME"] = r"C:/Users/陈一心/hadoop"
os.environ["PATH"] += r";C:/Users/陈一心/hadoop/bin"

# 1. SparkSession
spark = SparkSession.builder \
    .appName("FraudDetectionStreaming") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .getOrCreate()

# 2. Kafka Topic
kafka_bootstrap = "localhost:9092"
topic = "transactions"

# 3. Schema
schema = StructType()
for i in range(1, 29):
    schema = schema.add(f"V{i}", DoubleType())
schema = schema.add("Amount", DoubleType())

# 4. 模型 + Redis 初始化
model, threshold = joblib.load("fraud_detector.pkl")
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)


# 6. Streaming 处理函数
import rules  # 自动加载规则系统 + rule_results

def process_batch(df, _):
    print("⚡收到一批数据")
    df.show(truncate=False)
    records = df.collect()

    for row in records:
        data = row.asDict()
        msg_id = str(hash(str(data)))
        data["__id__"] = msg_id  # 加入追踪字段


        # 模型预测（去除 __id__ 字段）
        features = [float(data[k]) for k in data if k != "__id__"]
        vector = np.array([features])
        score = float(model.predict_proba(vector)[:, 1][0])

        # 清洗数据供规则引擎使用
        try:
            clean_data = {k: float(v) for k, v in data.items() if k != "__id__"}
            clean_data["__id__"] = msg_id
            assert_fact("fraud", clean_data)
        except Exception as e:
            print(f"⚠️ 规则引擎抛出异常: {repr(e)}")
            traceback.print_exc()

        # 获取规则结果
        rule_result = rules.get_rule_result(msg_id)
        hit = rule_result is not None

        if hit:
            print(f"🎯 命中规则: {rule_result}")

        # 融合判断
        final_flag = int(score >= threshold or hit)

        redis_client.set(f"tx:{msg_id}", json.dumps({
            "score": round(score, 4),
            "rule_hit": hit,
            "rule": rule_result['rule'] if hit else None,
            "risk": rule_result['risk'] if hit else None,
            "final_flag": final_flag
        }))
        print(f"✅ 写入 Redis → key: tx:{msg_id}, value: {{score: {score:.4f}, rule_hit: {hit}, final_flag: {final_flag}}}")


# 7. 编解析 Kafka 流
kafka_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", kafka_bootstrap) \
    .option("subscribe", topic) \
    .load()

json_df = kafka_df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

query = json_df.writeStream \
    .outputMode("append") \
    .foreachBatch(process_batch) \
    .start()

print("🔥 Spark 流式任务已启动...")
query.awaitTermination()
