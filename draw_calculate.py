import pandas as pd

# Load the dataset
data=pd.read_csv("C:\\Users/陈一心/.kaggle/archive/creditcard.csv")

# Show the first few rows of the dataset to understand its structure
data.head()

# Check for missing values in the dataset
missing_values = data.isnull().sum()
missing_values_percentage = (missing_values / len(data)) * 100

# Display the missing values and their percentages
missing_values, missing_values_percentage

import seaborn as sns
import matplotlib.pyplot as plt

# Compute the correlation matrix
correlation_matrix = data.corr()

# Plot the heatmap for correlation
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()

from sklearn.decomposition import PCA

# Drop non-numeric columns (Time, Amount, Class)
X = data.drop(columns=['Time', 'Amount', 'Class'])

# Apply PCA to reduce dimensionality, we will retain 95% of variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X)

# Create a DataFrame with PCA components
pca_df = pd.DataFrame(X_pca)

# Show the explained variance ratio to understand how much variance is captured by the principal components
explained_variance_ratio = pca.explained_variance_ratio_
explained_variance_ratio.sum(), pca_df.head()

# Adding a simple derived feature: Transaction frequency over a sliding window
# We will calculate the transaction frequency per 10-minute window.

# Adding a column to represent the transaction count in a rolling window (10-time units in this case)
data['Transaction_Frequency'] = data.groupby('Time')['Time'].transform('count')

# Show the new feature and sample rows
data[['Time', 'Transaction_Frequency']].head()

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, make_scorer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Prepare the data (exclude 'Time', 'Amount', and 'Class')
X = data.drop(columns=['Time', 'Amount', 'Class'])
y = data['Class']

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(solver='liblinear'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=0),
    'XGBoost': XGBClassifier(eval_metric='logloss')
}

# Initialize StratifiedKFold for cross-validation (3 folds)
kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)


# Function to calculate and evaluate metrics (Precision, Recall, F1-Score, and AUC-ROC)
def evaluate_metrics_from_cv(model, X, y, kf):
    # Initialize arrays to store predictions and probabilities
    y_pred = np.zeros_like(y)
    y_pred_proba = np.zeros((y.shape[0], 2))

    # Cross-validation loop
    for train_idx, test_idx in kf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)  # Fit model on training data
        y_pred[test_idx] = model.predict(X_test)  # Get predictions
        y_pred_proba[test_idx] = model.predict_proba(X_test)  # Get probabilities

    # Calculate precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary')

    # Calculate AUC-ROC score
    auc_roc = roc_auc_score(y, y_pred_proba[:, 1])

    # Get the ROC curve for plotting
    fpr, tpr, _ = roc_curve(y, y_pred_proba[:, 1])

    # Plot the ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'{model.__class__.__name__} (AUC = {auc_roc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guess')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.show()

    # Return the evaluation metrics
    return precision, recall, f1, auc_roc


# Evaluate each model and display the metrics
for model_name, model in models.items():
    print(f"Evaluating {model_name}...")
    precision, recall, f1, auc_roc = evaluate_metrics_from_cv(model, X, y, kf)
    print(
        f"{model_name} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, AUC-ROC: {auc_roc:.4f}")