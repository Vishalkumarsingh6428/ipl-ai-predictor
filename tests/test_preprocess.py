import pytest
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")
RAW_DIR  = Path("data/raw")


@pytest.fixture
def matches():
    return pd.read_csv(DATA_DIR / "matches_clean.csv")


@pytest.fixture
def deliveries():
    return pd.read_csv(DATA_DIR / "deliveries_clean.csv")


@pytest.fixture
def features():
    return pd.read_csv(DATA_DIR / "match_features.csv")


# --- Raw file checks ---

def test_raw_files_exist():
    assert (RAW_DIR / "matches.csv").exists(), "matches.csv not found"
    assert (RAW_DIR / "deliveries.csv").exists(), "deliveries.csv not found"


def test_processed_files_exist():
    assert (DATA_DIR / "matches_clean.csv").exists(), \
        "matches_clean.csv not found"
    assert (DATA_DIR / "deliveries_clean.csv").exists(), \
        "deliveries_clean.csv not found"
    assert (DATA_DIR / "match_features.csv").exists(), \
        "match_features.csv not found"


# --- Matches clean checks ---

def test_matches_no_null_winner(matches):
    # A small number of abandoned/no-result matches may have null winners
    null_count = matches["winner"].isnull().sum()
    assert null_count <= 10, \
        f"Too many null winners: {null_count} (expected <= 10)"


def test_matches_has_required_columns(matches):
    required = ["id", "winner", "target_runs", "venue", "city",
                "toss_winner", "toss_decision", "date"]
    for col in required:
        assert col in matches.columns, f"Missing column in matches: {col}"


def test_matches_date_parseable(matches):
    try:
        pd.to_datetime(matches["date"])
    except Exception:
        pytest.fail("Date column in matches_clean.csv is not parseable")


# --- Deliveries clean checks ---

def test_deliveries_has_required_columns(deliveries):
    required = [
        "match_id", "inning", "batting_team", "bowling_team",
        "over", "ball", "total_runs", "is_wicket"
    ]
    for col in required:
        assert col in deliveries.columns, \
            f"Missing column in deliveries: {col}"


def test_deliveries_inning_values(deliveries):
    # Innings 5 and 6 are Super Overs — valid in this dataset
    assert set(deliveries["inning"].unique()).issubset({1, 2, 3, 4, 5, 6}), \
        "Inning values must be between 1 and 6"


def test_deliveries_runs_non_negative(deliveries):
    assert (deliveries["total_runs"] >= 0).all(), \
        "total_runs must be non-negative"


def test_deliveries_is_wicket_binary(deliveries):
    assert set(deliveries["is_wicket"].unique()).issubset({0, 1}), \
        "is_wicket must be 0 or 1"


# --- Feature dataset join checks ---

def test_features_match_ids_in_matches(features, matches):
    feature_ids = set(features["match_id"].unique())
    match_ids   = set(matches["id"].unique())
    orphans     = feature_ids - match_ids
    assert len(orphans) == 0, \
        f"Feature rows with match_ids not in matches: {orphans}"


def test_features_row_count_reasonable(features):
    assert len(features) >= 50000, \
        f"Expected at least 50k rows, got {len(features)}"