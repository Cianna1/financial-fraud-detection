# train_model.py
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import joblib

# 1. 载入数据
data = pd.read_csv("C:\\Users\\陈一心\\.kaggle\\archive\\creditcard.csv")
X = data.drop(columns=['Time', 'Class'])
y = data['Class']

# 2. 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. 训练 XGBoost 模型（示例参数，可根据需要调整）
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(len(y) - sum(y)) / sum(y),
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)

# 4. 在测试集上调最佳阈值以最大化 F1
y_pred_prob = model.predict_proba(X_test)[:, 1]
best_threshold, best_f1 = 0.5, 0
for t in np.arange(0.1, 0.9, 0.01):
    y_pred = (y_pred_prob >= t).astype(int)
    f1 = f1_score(y_test, y_pred)
    if f1 > best_f1:
        best_f1, best_threshold = f1, t

print(f"Best threshold = {best_threshold:.2f}, F1 = {best_f1:.4f}")

# 5. 保存模型和阈值
joblib.dump((model, float(best_threshold)), ".ipynb_checkpoints/fraud_detector.pkl")
