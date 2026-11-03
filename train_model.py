import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib  # <-- use joblib instead of pickle
import os

# Create models folder if it doesn't exist
os.makedirs('models', exist_ok=True)

# Sample Data
data = {
    'WBC': [4500, 2800, 8000, 3100, 5000, 2900],
    'Platelets': [150000, 80000, 250000, 90000, 200000, 75000],
    'Hematocrit': [42, 50, 40, 48, 41, 52],
    'Hemoglobin': [14, 16, 13, 15, 14, 17],
    'Result': [0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

X = df[['WBC', 'Platelets', 'Hematocrit', 'Hemoglobin']]
y = df['Result']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model with joblib
joblib.dump(model, 'models/dengue_model.pkl')

print("Model trained and saved successfully with joblib!")