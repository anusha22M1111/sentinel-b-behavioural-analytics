# model_training.py

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss
)

from preprocessing import prepare_data
from clustering import perform_clustering


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "brier_score": brier_score_loss(y_test, y_prob)
    }

    return metrics


def train_models(df):

    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=150, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000)
    }

    results = {}
    best_model = None
    best_auc = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_model = model

    # Clustering analysis
    kmeans, clusters, silhouette = perform_clustering(X_train)
    results["clustering_silhouette_score"] = silhouette

    print("Training completed.")
    print("Best ROC-AUC:", best_auc)

    return best_model, results


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent

    data_path = BASE_DIR / "data" / "synthetic_student_behavior.csv"
    model_path = BASE_DIR / "models" / "trained_model.pkl"
    report_path = BASE_DIR / "reports" / "model_metrics.json"

    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    best_model, results = train_models(df)

    # Save best model
    joblib.dump(best_model, model_path)

    # Save metrics
    with open(report_path, "w") as f:
        json.dump(results, f, indent=4)

    print("Training completed successfully.")