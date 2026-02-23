import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna()
    return df

def encode_labels(df):
    le = LabelEncoder()
    df['Label'] = le.fit_transform(df['Label'])
    return df, le

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler