import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.metrics import f1_score
# -------------------------
# Load data & EDA
# -------------------------
data = pd.read_csv("C:\\Users/陈一心/.kaggle/archive/creditcard.csv")

# Heatmap of correlation
plt.figure(figsize=(14, 10))
sns.heatmap(data.corr(), cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()

# -------------------------
# Add: Histogram & Boxplot of Transaction Amount
# -------------------------
plt.figure(figsize=(14, 5))

# Left: Histogram
plt.subplot(1, 2, 1)
plt.hist(data['Amount'], bins=100, color='steelblue', edgecolor='black')
plt.title('Distribution of Transaction Amount')
plt.xlabel('Transaction Amount')
plt.ylabel('Frequency')

# Right: Boxplot
plt.subplot(1, 2, 2)
sns.boxplot(x='Class', y='Amount', data=data, showfliers=True)
plt.title('Box Plot of Transaction Amount by Class')
plt.xlabel('Class (0: Normal, 1: Fraud)')
plt.ylabel('Transaction Amount')

plt.tight_layout()
plt.show()

# -------------------------
# Add derived feature: Transaction Frequency
# -------------------------
data['Transaction_Frequency'] = data.groupby('Time')['Time'].transform('count')
# -------------------------
# Prepare features and labels
# -------------------------
X = data.drop(columns=['Time', 'Amount', 'Class'])  # 'Amount' not normalized, exclude it
X['Transaction_Frequency'] = data['Transaction_Frequency']
y = data['Class']

# Train/test split to prevent leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# -------------------------
# Define Models & Pipelines
# -------------------------
models = {
    'Logistic Regression': LogisticRegression(solver='liblinear'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=0),
    'XGBoost': XGBClassifier(eval_metric='logloss', use_label_encoder=False)
}

# Unified pipeline with scaler, PCA, SMOTE, and model
def create_pipeline(model):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])

# -------------------------
# Evaluation function
# -------------------------

# 添加阈值搜索函数
def find_best_threshold(y_true, y_prob, metric=f1_score):
    thresholds = np.linspace(0.1, 0.99, 50)
    best_thresh = 0.5
    best_score = -1

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        score = metric(y_true, y_pred)
        if score > best_score:
            best_score = score
            best_thresh = thresh

    return best_thresh, best_score

# 主评估函数（替换原来的）
def evaluate_metrics_from_cv(pipeline, X, y, kf):
    y_pred_proba = np.zeros((y.shape[0], 2))

    for train_idx, test_idx in kf.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        pipeline.fit(X_tr, y_tr)
        y_pred_proba[test_idx] = pipeline.predict_proba(X_te)

    # 概率输出完成，执行阈值搜索
    best_thresh, best_f1 = find_best_threshold(y, y_pred_proba[:, 1])

    # 使用最佳阈值生成最终预测
    y_pred = (y_pred_proba[:, 1] >= best_thresh).astype(int)

    # 评估指标
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')
    auc_roc = roc_auc_score(y, y_pred_proba[:, 1])
    fpr, tpr, _ = roc_curve(y, y_pred_proba[:, 1])

    # ROC 曲线绘制
    plt.figure(figsize=(10, 6))
    model_name = pipeline.named_steps["model"].__class__.__name__
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_roc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guess')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

    # 输出最佳阈值信息
    print(f"Best Threshold: {best_thresh:.2f} | F1 at best threshold: {best_f1:.4f}")

    return precision, recall, f1, auc_roc


# -------------------------
# Run evaluation for each model
# -------------------------
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for model_name, model in models.items():
    print(f"\nEvaluating {model_name}...")
    pipeline = create_pipeline(model)
    precision, recall, f1, auc_roc = evaluate_metrics_from_cv(pipeline, X_train, y_train, kf)
    print(f"{model_name} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, AUC-ROC: {auc_roc:.4f}")

import joblib

# -------------------------
# 训练最终模型（以 XGBoost 为主力模型）
# -------------------------
final_model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
final_pipeline = create_pipeline(final_model)
final_pipeline.fit(X_train, y_train)

# 在整个训练集上获取概率预测并找最佳阈值
y_train_proba = final_pipeline.predict_proba(X_train)[:, 1]
final_best_threshold, final_best_f1 = find_best_threshold(y_train, y_train_proba)

# 输出确认信息
print(f"\n[Final Model Saved]")
print(f"Final Best Threshold: {final_best_threshold:.2f} | F1: {final_best_f1:.4f}")

# -------------------------
# 序列化保存模型及最佳阈值
# -------------------------
joblib.dump(final_pipeline, 'xgboost_pipeline.pkl')
joblib.dump(final_best_threshold, 'xgboost_threshold.pkl')
