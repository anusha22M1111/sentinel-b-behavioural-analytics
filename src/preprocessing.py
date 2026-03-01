# preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "logins",
    "delay",
    "attendance",
    "sentiment",
    "night_activity_ratio",
    "engagement_velocity",
    "engagement_acceleration",
    "attendance_trend",
    "sentiment_drift",
    "volatility_index",
    "burnout_risk_score"
]


def prepare_data(df):

    df = df.copy()

    # Target variable
    y = (df["risk_level"] == "High").astype(int)

    X = df[FEATURE_COLUMNS]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler