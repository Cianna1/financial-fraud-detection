import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


import tensorflow as tf
import random
import os
# 固定种子
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
# 读取数据
data = pd.read_csv("C:\\Users\\陈一心\\.kaggle\\archive\\creditcard.csv")
X = data.drop(columns=['Time', 'Class'])
y = data['Class']

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# 定义 MLP 模型
model = Sequential([
    Dense(32, input_dim=X_train.shape[1], activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# 编译模型
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['AUC'])

# 训练模型
model.fit(X_train, y_train, epochs=2, batch_size=256, verbose=1)

# 预测并评估 AUC
y_pred_prob = model.predict(X_test).ravel()
auc_score = roc_auc_score(y_test, y_pred_prob)
print("MLP AUC:", auc_score)

# ✅ 保存模型
model.save("mlp_model.keras")