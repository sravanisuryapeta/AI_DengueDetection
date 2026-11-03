import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# --- Step 1: Generate a larger synthetic dataset ---
np.random.seed(42)
num_samples = 1000  # You can increase this

# Generate CBC features within realistic ranges
WBC = np.random.randint(3000, 12000, size=num_samples)       # White Blood Cell count
Platelets = np.random.randint(50000, 350000, size=num_samples)  # Platelet count
Hematocrit = np.random.randint(30, 55, size=num_samples)     # %
Hemoglobin = np.random.randint(10, 18, size=num_samples)     # g/dL

# Assign dengue risk based on simple logic (just for demo)
# Low Platelets (<100k) or high WBC (>10000) → more likely dengue
Result = np.where((Platelets < 100000) | (WBC > 10000), 1, 0)

data = {
    'WBC': WBC,
    'Platelets': Platelets,
    'Hematocrit': Hematocrit,
    'Hemoglobin': Hemoglobin,
    'Result': Result
}

df = pd.DataFrame(data)

# --- Step 2: Train a Random Forest Model ---
X = df[['WBC', 'Platelets', 'Hematocrit', 'Hemoglobin']]
y = df['Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Step 3: Save the model & dataset ---
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/dengue_model.pkl')  # save model
df.to_pickle('dengue_dataset.pkl')             # save dataset for app

print("Synthetic dataset and model created successfully!")
print(f"Total records: {len(df)}")
print(f"Dengue positive cases: {df['Result'].sum()}")
print(f"Avg Platelet Count: {df['Platelets'].mean():.0f}")