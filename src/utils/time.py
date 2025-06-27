import pandas as pd
import time
import requests

df = pd.read_csv("C:\\Users\\陈一心\\.kaggle\\archive\\creditcard.csv").sample(n=20, random_state=42)

features = [col for col in df.columns if col.startswith("V")] + ["Amount"]

for idx, row in df.iterrows():
    payload = {col: float(row[col]) for col in features}
    try:
        res = requests.post("http://localhost:8000/predict", json=payload)
        result = res.json()
        if "fraud_probability" in result:
            print(f"[实时判定] fraud: {result['is_fraud']} | 概率: {result['fraud_probability']:.4f} → 风险等级: {result['risk_level']}")
        else:
            print("请求失败：响应中缺字段", result)
    except Exception as e:
        print("请求失败：", e)
    time.sleep(1)

