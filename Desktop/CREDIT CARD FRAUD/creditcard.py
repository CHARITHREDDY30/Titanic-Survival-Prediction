# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# Step 2: Load the Dataset
data_path = r"C:\Users\Charith\Desktop\CREDIT CARD FRAUD\creditcard.csv"
df = pd.read_csv(data_path)

# Step 3: Explore the Dataset
print("Dataset Shape:", df.shape)
print("Class Distribution:\n", df['Class'].value_counts())

# Step 4: Data Preprocessing
df = df.drop(columns=['Time'])  # Drop Time column
df['Amount'] = StandardScaler().fit_transform(df[['Amount']])  # Normalize Amount

# Step 5: Prepare Features and Labels
X = df.drop('Class', axis=1)
y = df['Class']

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    stratify=y, 
                                                    random_state=42)

# Step 7: Handle Class Imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("After SMOTE class distribution:\n", pd.Series(y_train_res).value_counts())

# Step 8: Train Models

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_res, y_train_res)
y_pred_lr = lr.predict(X_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_res, y_train_res)
y_pred_rf = rf.predict(X_test)

# Step 9: Evaluate Models
print("\n--- Logistic Regression Evaluation ---")
print(classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

print("\n--- Random Forest Evaluation ---")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
