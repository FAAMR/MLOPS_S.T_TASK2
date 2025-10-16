import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import json
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load data
df = pd.read_csv(r'C:\Users\FARIDA\Desktop\TASK\MLOPS_S.T_TASK2\data\heart_raw.csv')

# Encode categorical columns
categorical_cols = ['Sex', 'ChestPainType', 'ST_Slope', 'ExerciseAngina', 'RestingECG']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Features and target
X = df_encoded.drop('HeartDisease', axis=1)
y = df_encoded['HeartDisease']

# Train model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X, y)
preds = model.predict(X)

# Ensure the MY_Data directory exists
os.makedirs('MY_Data', exist_ok=True)

# Save metrics in MY_Data
acc = accuracy_score(y, preds)
with open('MY_Data/metrics.json', 'w') as f:
    json.dump({'accuracy': acc}, f)

# Generate and save confusion matrix plot in MY_Data
cm = confusion_matrix(y, preds, labels=model.classes_)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('MY_Data/confusion_matrix.png')
plt.close()

print("Training done. Metrics and confusion matrix saved in MY_Data folder.")
