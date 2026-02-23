import pandas as pd
from utils import scale_features
from scorer import balance_data
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib

print("Loading UNSW Dataset...")

# Load dataset
df = pd.read_csv("../data/UNSW_NB15_training-set.csv")
df = df.dropna()

# Target column
y = df['label']
X = df.drop(['label'], axis=1)

# Convert categorical to numerical
X = pd.get_dummies(X)

# 🔥 SAVE TRAINING FEATURE COLUMNS (VERY IMPORTANT)
columns = X.columns
joblib.dump(columns, "../models/columns.pkl")

print("Columns Saved!")

# Scale Features
X_scaled, scaler = scale_features(X)

# Handle Class Imbalance
X_bal, y_bal = balance_data(X_scaled, y)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2
)

# Train Model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Save Model & Scaler
joblib.dump(model, "../models/ids_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")

print("Model Trained with UNSW-NB15 & Saved!")