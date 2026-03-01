# data_generator.py

import numpy as np
import pandas as pd
from config import *
from state_definitions import generate_state_features
from feature_engineering import compute_temporal_features
from risk_scoring import calculate_risk_score


def next_state(current_state):
    probs = TRANSITION_MATRIX[current_state]
    return np.random.choice(list(STATES.keys()), p=probs)


def generate_dataset():
    np.random.seed(RANDOM_SEED)

    records = []

    for student in range(N_STUDENTS):
        state = np.random.choice(list(STATES.keys()))

        for week in range(N_WEEKS):
            logins, delay, attendance, sentiment, night_ratio = generate_state_features(state)

            records.append([
                student,
                week,
                state,
                logins,
                delay,
                attendance,
                sentiment,
                night_ratio
            ])

            state = next_state(state)

    df = pd.DataFrame(records, columns=[
        "student_id",
        "week",
        "state",
        "logins",
        "delay",
        "attendance",
        "sentiment",
        "night_activity_ratio"
    ])

    df = compute_temporal_features(df)
    df = calculate_risk_score(df)

    df["dropout_probability"] = 1 / (
        1 + np.exp(-0.05 * (df["burnout_risk_score"] - 50))
    )

    return df


if __name__ == "__main__":
    dataset = generate_dataset()
    dataset.to_csv("D:\mtech-cse-ba\projects\sentinel-b\synthetic_student_behavior.csv", index=False)
    print("Dataset generated successfully.")