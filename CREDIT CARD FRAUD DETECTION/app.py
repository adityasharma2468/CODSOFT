import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.preprocessing import StandardScaler

# -------------
# Dataset Load 
# -------------
df = pd.read_csv(
    'CREDIT CARD FRAUD DETECTION/Credi Card.csv'
)
print("Dataset Loaded!")
print("Shape:", df.shape)
print("\nClass Distribution:")
print(df['Class'].value_counts())
print(f"\nFraud %: {df['Class'].mean()*100:.2f}%")

# -------------
# Data Explore 
# -------------
print("\n=== MISSING VALUES ===")
print(df.isnull().sum().sum(), 
      "missing values")

# ----------------
# Data Preprocess 
# ----------------

# Amount Scale
scaler = StandardScaler()
df['Amount_Scaled'] = scaler.fit_transform(
    df[['Amount']]
)
df['Time_Scaled'] = scaler.fit_transform(
    df[['Time']]
)

# Original columns drop 
df.drop(['Amount', 'Time'], 
        axis=1, inplace=True)

print("\nData Preprocessed!")

# ----------------------
# Class Imbalance Handle
# ----------------------

# Undersampling 
fraud = df[df['Class'] == 1]
genuine = df[df['Class'] == 0].sample(
    n=len(fraud), 
    random_state=42
)

# Balance dataset 
df_balanced = pd.concat(
    [fraud, genuine]
).sample(frac=1, random_state=42)

print("\n Class Balance Done!")
print("Fraud:", len(fraud))
print("Genuine:", len(genuine))
print("Total:", len(df_balanced))

# ----------------
# Features Select 
# ----------------
X = df_balanced.drop('Class', axis=1)
y = df_balanced['Class']

# ----------------
# Train Test Split
# ----------------
X_train, X_test, y_train, y_test = \
    train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

print("\n Data Split Done!")
print("Train:", X_train.shape)
print("Test:", X_test.shape)

# ------------
# Model Train 
# ------------

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print("\nModels Trained!")
print(f"Logistic Regression: {lr_acc*100:.2f}%")
print(f"Random Forest      : {rf_acc*100:.2f}%")

# --------
# Evaluate 
# --------
print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(
    y_test, rf_pred,
    target_names=['Genuine', 'Fraud']
))

# -------------
# Visualization :--
# -------------
fig, axes = plt.subplots(2, 2,
                          figsize=(12, 10))
fig.suptitle(
    'Credit Card Fraud Detection',
    fontsize=16
)

# Plot 1 - Class Distribution (Visualization)
df_balanced['Class'].value_counts().plot(
    kind='bar',
    ax=axes[0,0],
    color=['#FF9C04', '#BC1880']
)
axes[0,0].set_title('Class Distribution')
axes[0,0].set_xticklabels(
    ['Genuine', 'Fraud'],
    rotation=0
)

# Plot 2 - Confusion Matrix (Visualization)
cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(
    cm, annot=True, fmt='d',
    ax=axes[0,1],
    cmap='Blues',
    xticklabels=['Genuine', 'Fraud'],
    yticklabels=['Genuine', 'Fraud']
)
axes[0,1].set_title('Confusion Matrix')

# Plot 3 - Amount Distribution (Visualization)
sns.histplot(
    df_balanced['Amount_Scaled'],
    bins=30,
    ax=axes[1,0],
    color='#1BABD5'
)
axes[1,0].set_title(
    'Transaction Amount Distribution'
)

# Plot 4 - Model Comparison (Visualization)
models = ['Logistic\nRegression', 
          'Random\nForest']
accuracies = [lr_acc*100, rf_acc*100]
axes[1,1].bar(
    models, accuracies,
    color=['#323D2C', '#68765C']
)
axes[1,1].set_title('Model Accuracy Comparison')
axes[1,1].set_ylabel('Accuracy %')
axes[1,1].set_ylim([0, 100])
for i, v in enumerate(accuracies):
    axes[1,1].text(
        i, v+1,
        f'{v:.2f}%',
        ha='center'
    )

plt.tight_layout()
plt.savefig(
    'CREDIT CARD FRAUD DETECTION/fraud_analysis.png'
)
plt.show()
