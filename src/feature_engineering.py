# feature_engineering.py
import numpy as np
import pandas as pd

def compute_temporal_features(df):
    """
    Compute velocity, acceleration and volatility metrics.
    """

    df = df.sort_values(["student_id", "week"])

    df["engagement_velocity"] = df.groupby("student_id")["logins"].diff().fillna(0)
    df["engagement_acceleration"] = df.groupby("student_id")["engagement_velocity"].diff().fillna(0)

    df["attendance_trend"] = df.groupby("student_id")["attendance"].diff().fillna(0)

    df["sentiment_drift"] = df.groupby("student_id")["sentiment"].diff().fillna(0)

    df["volatility_index"] = (
        df.groupby("student_id")["logins"]
        .rolling(3)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )

    return df