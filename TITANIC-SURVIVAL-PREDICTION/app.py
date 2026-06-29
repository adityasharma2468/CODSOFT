import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ---------------------
# STEP 1 - Dataset Load 
# ---------------------

df = pd.read_csv('TITANIC SURVIVAL PREDICTION/Titanic-Dataset.csv')
print(" Dataset Loaded!")
print("Shape:", df.shape)

# ---------------------
# STEP 2 - Data Clean 
# ---------------------
# Age missing fill 

df['Age'].fillna(df['Age'].median(), 
                 inplace=True)

# Embarked missing fill 
df['Embarked'].fillna(
    df['Embarked'].mode()[0], 
    inplace=True)

# Cabin drop 

df.drop(['Cabin', 'Name', 
         'Ticket', 'PassengerId'], 
         axis=1, inplace=True)

# Gender encode 

df['Sex'] = df['Sex'].map(
    {'male': 0, 'female': 1})

# Embarked encode 

df['Embarked'] = df['Embarked'].map(
    {'S': 0, 'C': 1, 'Q': 2})

#  NaN check 

print("\n=== MISSING VALUES AFTER CLEAN ===")
print(df.isnull().sum())

# Remaining NaN drop 

df.dropna(inplace=True)
print(" Data Cleaned!")
print("Shape after cleaning:", df.shape)

# -------------------------
# STEP 3 - Features Select 
# -------------------------

X = df[['Pclass', 'Sex', 'Age',
        'SibSp', 'Parch', 'Fare',
        'Embarked']]
y = df['Survived']

print("\n Features Selected!")
print("X shape:", X.shape)

# -------------------------
# STEP 4 - Train Test Split
# -------------------------

X_train, X_test, y_train, y_test = \
    train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )
print(" Data Split Done!")
print("Train:", X_train.shape)
print("Test:", X_test.shape)

# ---------------------
# STEP 5 - Model Train 
# ---------------------
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

print("\n Models Trained!")
print(f"Logistic Regression: {lr_acc*100:.2f}%")
print(f"Random Forest      : {rf_acc*100:.2f}%")

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, rf_pred))

# ----------------------
# STEP 6 - Visualization
# ----------------------

fig, axes = plt.subplots(2, 2, 
                          figsize=(12, 10))
fig.suptitle(
    'Titanic Survival Analysis',
    fontsize=16)

# Plot 1
sns.countplot(x='Survived', data=df,
              ax=axes[0,0],
              palette=['red','green'])
axes[0,0].set_title('Survival Count')

# Plot 2
sns.countplot(x='Survived', hue='Sex',
              data=df, ax=axes[0,1],
              palette=['blue','pink'])
axes[0,1].set_title('Survival by Gender')

# Plot 3
sns.countplot(x='Pclass', hue='Survived',
              data=df, ax=axes[1,0],
              palette=['red','green'])
axes[1,0].set_title('Survival by Class')

# Plot 4
sns.histplot(df['Age'], bins=30,
             ax=axes[1,1], color='blue')
axes[1,1].set_title('Age Distribution')

plt.tight_layout()
plt.savefig(
    'TITANIC SURVIVAL PREDICTION/titanic_analysis.png')
plt.show()
