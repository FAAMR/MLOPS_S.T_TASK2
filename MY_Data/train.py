import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import json
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(r'C:\Users\FARIDA\Desktop\TASK\MLOPS_S.T_TASK2\data\heart_raw.csv')

# -------------------------------
# Handle missing values
# -------------------------------
df.replace('?', pd.NA, inplace=True)

# Fill numerical columns with median
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[num_cols] = df[num_cols].fillna(df[num_cols].median())

# Fill categorical columns with mode
categorical_cols = ['Sex', 'ChestPainType', 'ST_Slope', 'ExerciseAngina', 'RestingECG']
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# -------------------------------
# Encode categorical columns
# -------------------------------
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# -------------------------------
# Features and target
# -------------------------------
X = df_encoded.drop('HeartDisease', axis=1)
y = df_encoded['HeartDisease']

# -------------------------------
# Train/test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Train Random Forest model
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
preds = model.predict(X_test)

# -------------------------------
# Ensure MY_Data directory exists
# -------------------------------
os.makedirs('MY_Data', exist_ok=True)

# -------------------------------
# Save metrics
# -------------------------------
acc = accuracy_score(y_test, preds)
with open('MY_Data/metrics.json', 'w') as f:
    json.dump({'accuracy': acc}, f)

# -------------------------------
# Save confusion matrix plot
# -------------------------------
cm = confusion_matrix(y_test, preds, labels=model.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('MY_Data/confusion_matrix.png')
plt.close()

# -------------------------------
# Save trained model
# -------------------------------
joblib.dump(model, 'MY_Data/model.pkl')

print("Training done. Metrics, confusion matrix, and model saved in MY_Data folder.")
