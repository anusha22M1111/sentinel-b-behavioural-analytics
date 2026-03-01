# risk_scoring.py
import numpy as np
import pandas as pd
def calculate_risk_score(df):
    """
    Custom behavioural risk formulation.
    """

    score = (
        0.25 * df["engagement_acceleration"].abs() +
        0.20 * df["attendance_trend"].abs() +
        0.20 * df["sentiment_drift"].abs() +
        0.15 * df["delay"] +
        0.20 * df["volatility_index"]
    )

    score = 100 * (score - score.min()) / (score.max() - score.min() + 1e-6)

    df["burnout_risk_score"] = score

    df["risk_level"] = pd.cut(
        score,
        bins=[0, 40, 70, 100],
        labels=["Low", "Medium", "High"]
    )

    return df