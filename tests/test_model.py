import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

MODELS_DIR = Path("models")
DATA_DIR   = Path("data/processed")


@pytest.fixture
def pipeline():
    return joblib.load(MODELS_DIR / "xgboost.pkl")


@pytest.fixture
def sample_input():
    return pd.DataFrame([{
        "batting_team":      "Mumbai Indians",
        "bowling_team":      "Chennai Super Kings",
        "venue":             "Wankhede Stadium",
        "city":              "Mumbai",
        "toss_winner":       "Mumbai Indians",
        "toss_decision":     "field",
        "batting_team_form": 0.6,
        "bowling_team_form": 0.4,
        "runs_left":         45,
        "balls_left":        30,
        "wickets_left":      5,
        "crr":               8.5,
        "rrr":               9.0
    }])


def test_model_file_exists():
    assert (MODELS_DIR / "xgboost.pkl").exists(), \
        "xgboost.pkl not found in models/"


def test_shap_explainer_exists():
    assert (MODELS_DIR / "shap_explainer.pkl").exists(), \
        "shap_explainer.pkl not found in models/"


def test_feature_names_exist():
    assert (MODELS_DIR / "feature_names.pkl").exists(), \
        "feature_names.pkl not found in models/"


def test_model_predicts(pipeline, sample_input):
    pred = pipeline.predict(sample_input)
    assert pred.shape == (1,), "Prediction shape mismatch"
    assert pred[0] in [0, 1], "Prediction must be 0 or 1"


def test_model_predict_proba(pipeline, sample_input):
    proba = pipeline.predict_proba(sample_input)
    assert proba.shape == (1, 2), "predict_proba shape must be (1, 2)"
    assert abs(proba[0].sum() - 1.0) < 1e-5, \
        "Probabilities must sum to 1"


def test_win_probability_range(pipeline, sample_input):
    proba = pipeline.predict_proba(sample_input)[0][1]
    assert 0.0 <= proba <= 1.0, \
        f"Win probability out of range: {proba}"


def test_model_bulk_predictions(pipeline):
    df = pd.read_csv(DATA_DIR / "match_features.csv")
    X  = df.drop(["result", "match_id"], axis=1).head(100)
    preds = pipeline.predict(X)
    assert len(preds) == 100, "Expected 100 predictions"
    assert set(preds).issubset({0, 1}), "All predictions must be 0 or 1"


def test_model_accuracy_threshold(pipeline):
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GroupShuffleSplit

    df     = pd.read_csv(DATA_DIR / "match_features.csv")
    groups = df["match_id"]
    X      = df.drop(["result", "match_id"], axis=1)
    y      = df["result"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(gss.split(X, y, groups))

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    preds    = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    assert accuracy >= 0.70, \
        f"Model accuracy too low: {accuracy:.4f} (expected >= 0.70)"