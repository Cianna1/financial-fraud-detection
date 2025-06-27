# 1. 模型训练与调优（略）
# （此部分已完成，不再重复，重点在 FastAPI 集成）
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd

# 载入数据
data = pd.read_csv("C:\\Users\\\u9648\u4e00\u5fc3\\.kaggle\\archive\\creditcard.csv")
X = data.drop(columns=['Time', 'Class'])
y = data['Class']

# 分类器调优
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=5,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model.fit(X_train, y_train)

# 预测概率
y_pred_prob = model.predict_proba(X_test)[:, 1]

# 查找最佳阈值
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

# 最终预测与评估
y_pred_final = (y_pred_prob >= best_threshold).astype(int)
print("Precision:", precision_score(y_test, y_pred_final))
print("Recall:", recall_score(y_test, y_pred_final))
print("F1-Score:", f1_score(y_test, y_pred_final))
print("AUC-ROC:", roc_auc_score(y_test, y_pred_prob))

# 保存模型
import joblib
joblib.dump((model, best_threshold), '.ipynb_checkpoints/fraud_detector.pkl')
# 2. FastAPI + durable_rules + MLP 并行融合服务
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
import numpy as np
import joblib
import requests
from rules import get_rule_result
from durable.lang import post
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import os

# 初始化 FastAPI 应用
app = FastAPI()

# 限流器
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Prometheus metrics 集成
instrumentator = Instrumentator().instrument(app).expose(app)

# MLP 模型和预处理器载入
mlp_model = load_model("src/mlp_model.keras")
scaler = joblib.load("src/mlp_model_v2.pkl")  # 假设预处理器已保存

# 请求体结构
class Transaction(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float
    V8: float; V9: float; V10: float; V11: float; V12: float; V13: float; V14: float
    V15: float; V16: float; V17: float; V18: float; V19: float; V20: float; V21: float
    V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float
    Amount: float

# 重试封装（适用于 XGBoost）
@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def safe_predict(model, data):
    return model.predict_proba(data)[:, 1]

# 接口入口
@app.post("/predict")
@limiter.limit("10/minute")
def predict(transaction: Transaction, request: Request):
    # 加载 XGBoost 模型
    model, threshold = joblib.load(".ipynb_checkpoints/fraud_detector.pkl")
    input_array = np.array([list(transaction.dict().values())])

    # XGBoost 预测概率
    try:
        prob_xgb = float(safe_predict(model, input_array)[0])
    except RetryError:
        prob_xgb = 0.0

    # MLP 预测概率
    input_scaled = scaler.transform(input_array)
    prob_mlp = float(mlp_model.predict(input_scaled, verbose=0)[0][0])

    # 融合预测（加权平均）
    final_prob = 0.7 * prob_xgb + 0.3 * prob_mlp

    # 调用规则引擎
    transaction_dict = transaction.dict()
    transaction_dict['__id__'] = str(np.random.randint(1e6))
    post('fraud', transaction_dict)
    rule_result = get_rule_result(transaction_dict['__id__'])

    # 风险等级判定
    is_fraud = int(final_prob >= threshold)
    risk = 'high' if final_prob >= 0.8 else 'medium' if final_prob >= 0.5 else 'low'

    return {
        "is_fraud": is_fraud,
        "final_fraud_probability": round(final_prob, 6),
        "risk_level": risk,
        "prob_xgb": round(prob_xgb, 6),
        "prob_mlp": round(prob_mlp, 6),
        "rule_engine_result": rule_result
    }

# 启动服务
if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)

# 3. 示例请求（略）
# 3. 测试请求示例（也可放入 test_api.py）
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
