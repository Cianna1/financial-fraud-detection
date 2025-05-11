# app.py
import json
import numpy as np
from flask import Flask, request, jsonify
import joblib
import redis

# Flask 应用与 Redis 连接
app = Flask(__name__)
r = redis.Redis(host='localhost', port=6379, db=0)  # 修改 host/port 如有需要

# 风险等级划分函数
def risk_level(score: float) -> str:
    if score >= 0.8:
        return 'high'
    elif score >= 0.5:
        return 'medium'
    else:
        return 'low'

# 加载模型（可按需改成全局缓存）
model, threshold = joblib.load('fraud_detector.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json()
    if not payload:
        return jsonify({'error': 'Invalid JSON.'}), 400

    # 用 JSON 序列化作为缓存 key（顺序一定要排序）
    key = json.dumps(payload, sort_keys=True)
    # 如果缓存命中
    if r.exists(key):
        result = json.loads(r.get(key))
        result['cached'] = True
        return jsonify(result)

    try:
        # 构造输入数组
        features = np.array([list(payload.values())], dtype=float)  # shape (1,29)
        # 预测概率
        prob = float(model.predict_proba(features)[0][1])
        is_fraud = int(prob >= threshold)
        risk = risk_level(prob)

        result = {
            'is_fraud': is_fraud,
            'fraud_probability': round(prob, 4),
            'risk_level': risk,
            'cached': False
        }
        # 缓存 1 小时
        r.set(key, json.dumps(result), ex=3600)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
