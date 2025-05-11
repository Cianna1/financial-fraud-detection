# 1. 调优模型

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd

# 载入数据
data = pd.read_csv("C:\\Users\\陈一心\\.kaggle\\archive\\creditcard.csv")
X = data.drop(columns=['Time', 'Class'])
y = data['Class']

# 分类器调优
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=5,  # 根据负比比调整
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model.fit(X_train, y_train)

# 预测结果
y_pred_prob = model.predict_proba(X_test)[:, 1]

# 调整阈值
def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_f1 = 0
    best_thresh = 0.5
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh

best_threshold = find_best_threshold(y_test, y_pred_prob)

# 在最佳阈值下评估
y_pred_final = (y_pred_prob >= best_threshold).astype(int)
print("Precision:", precision_score(y_test, y_pred_final))
print("Recall:", recall_score(y_test, y_pred_final))
print("F1-Score:", f1_score(y_test, y_pred_final))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_prob))

# 2. 部署成API

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import joblib

# 保存模型
joblib.dump((model, best_threshold), 'fraud_detector.pkl')
print("模块已被成功加载")
app = FastAPI()

# 定义输入格式
class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


# 进行预测
@app.post("/predict")
def predict(transaction: Transaction):
    model, threshold = joblib.load('fraud_detector.pkl')
    data = np.array([list(transaction.dict().values())])
    prob_np = model.predict_proba(data)[:, 1][0]
    prob=float(prob_np)

    # 分级与返回
    is_fraud = int(prob >= threshold)  # 0 或 1
    if prob >= 0.8:
        risk = 'high'
    elif prob >= 0.5:
        risk = 'medium'
    else:
        risk = 'low'

    return { "is_fraud": is_fraud,"fraud_probability": prob, "risk_level": risk}

# 启动服务
if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)

# 3. 测试示例

# Python requests测试
import requests

url = "http://127.0.0.1:8000/predict"
data = {
    "V1": -1.23, "V2": 0.45, "V3": -0.89, "V4": 0.1, "V5": -1.2,
    "V6": 0.3, "V7": -0.1, "V8": 0.4, "V9": -0.7, "V10": 0.2,
    "V11": -0.4, "V12": 0.6, "V13": -1.1, "V14": 0.5, "V15": 0.3,
    "V16": -0.2, "V17": 0.8, "V18": -0.6, "V19": 0.9, "V20": -1.0,
    "V21": 0.2, "V22": -0.3, "V23": 0.7, "V24": -0.8, "V25": 0.1,
    "V26": -0.5, "V27": 0.3, "V28": -0.9, "Amount": 100.0
}

response = requests.post(url, json=data)
print(response.json())
