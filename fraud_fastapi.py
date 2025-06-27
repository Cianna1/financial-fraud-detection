import numpy as np
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from tenacity import retry, stop_after_attempt, wait_fixed, RetryError
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from joblib import load

# 加载完整模型（含预处理流程）和阈值
xgb_pipeline = load("xgboost_pipeline.pkl")
xgb_threshold = load("xgboost_threshold.pkl")

# 2. FastAPI 服务
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
from rules import get_rule_result, rule_results
from durable.lang import post

app = FastAPI()

# 限流器
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Prometheus metrics
instrumentator = Instrumentator().instrument(app).expose(app)

# 加载模型
mlp_model = load_model("src\mlp_model.keras")
scaler = load('src\mlp_scaler_v2.pkl')

class Transaction(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float; V6: float; V7: float
    V8: float; V9: float; V10: float; V11: float; V12: float; V13: float; V14: float
    V15: float; V16: float; V17: float; V18: float; V19: float; V20: float; V21: float
    V22: float; V23: float; V24: float; V25: float; V26: float; V27: float; V28: float
    Amount: float

@retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
def safe_predict(model, data):
    return model.predict_proba(data)[:, 1]

@app.post("/predict")
@limiter.limit("10/minute")
def predict(transaction: Transaction, request: Request):
    try:
        data = np.array([list(transaction.dict().values())])
        prob = float(safe_predict(xgb_pipeline, data)[0])
    except RetryError:
        return {"error": "模型预测失败，请稍后重试", "is_fraud": None}

    # 调用规则引擎
    tx_dict = transaction.dict()
    tx_dict['__id__'] = str(np.random.randint(1e6))
    post('fraud', tx_dict)
    rule_result = get_rule_result(tx_dict['__id__'])

    # 风险等级判断
    is_fraud = int(prob >= xgb_threshold)
    risk = 'high' if prob >= 0.8 else 'medium' if prob >= 0.5 else 'low'

    # MLP 概率
    scaled = scaler.transform(data)
    mlp_prob = float(mlp_model.predict(scaled)[0][0])

    return {
        "is_fraud": is_fraud,
        "fraud_probability": prob,
        "risk_level": risk,
        "rule_engine_result": rule_result,
        "mlp_probability": mlp_prob,
        "mlp_auc_threshold": 0.5
    }

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    uvicorn.run(app, host="0.0.0.0", port=8000)



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
