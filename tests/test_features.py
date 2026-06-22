import pytest
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/processed")


@pytest.fixture
def features():
    return pd.read_csv(DATA_DIR / "match_features.csv")


def test_required_columns_exist(features):
    required = [
        "match_id", "batting_team", "bowling_team", "venue", "city",
        "toss_winner", "toss_decision", "batting_team_form",
        "bowling_team_form", "runs_left", "balls_left",
        "wickets_left", "crr", "rrr", "result"
    ]
    for col in required:
        assert col in features.columns, f"Missing column: {col}"


def test_no_missing_values(features):
    assert features.isnull().sum().sum() == 0, "Dataset contains missing values"


def test_no_infinite_values(features):
    numeric = features.select_dtypes(include=[np.number])
    assert not np.isinf(numeric.values).any(), "Dataset contains infinite values"


def test_result_is_binary(features):
    assert set(features["result"].unique()).issubset({0, 1}), \
        "Result column must only contain 0 or 1"


def test_balls_left_range(features):
    assert (features["balls_left"] >= 0).all() and \
           (features["balls_left"] <= 132).all(), \
        "balls_left must be between 0 and 132"


def test_wickets_left_range(features):
    assert features["wickets_left"].between(0, 10).all(), \
        "wickets_left must be between 0 and 10"


def test_runs_left_non_negative(features):
    assert (features["runs_left"] >= 0).all(), \
        "runs_left must be non-negative"


def test_crr_non_negative(features):
    assert (features["crr"] >= 0).all(), \
        "crr (current run rate) must be non-negative"


def test_team_form_range(features):
    assert features["batting_team_form"].between(0, 1).all(), \
        "batting_team_form must be between 0 and 1"
    assert features["bowling_team_form"].between(0, 1).all(), \
        "bowling_team_form must be between 0 and 1"


def test_unique_matches(features):
    assert features["match_id"].nunique() >= 500, \
        "Expected at least 500 unique matches"


def test_balanced_results(features):
    win_rate = features["result"].mean()
    assert 0.3 <= win_rate <= 0.7, \
        f"Result distribution looks off: {win_rate:.2f} win rate"
