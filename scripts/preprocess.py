import pandas as pd
from pathlib import Path


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    matches = pd.read_csv(RAW_DIR / "matches.csv")
    deliveries = pd.read_csv(RAW_DIR / "deliveries.csv")

    return matches, deliveries


def clean_matches(matches):
    matches = matches.copy()

    matches.drop_duplicates(inplace=True)

    return matches


def clean_deliveries(deliveries):
    deliveries = deliveries.copy()

    deliveries.drop_duplicates(inplace=True)

    return deliveries


def save_data(matches, deliveries):
    matches.to_csv(
        PROCESSED_DIR / "matches_clean.csv",
        index=False
    )

    deliveries.to_csv(
        PROCESSED_DIR / "deliveries_clean.csv",
        index=False
    )


def main():

    matches, deliveries = load_data()

    matches = clean_matches(matches)
    deliveries = clean_deliveries(deliveries)

    save_data(matches, deliveries)

    print("Data preprocessing completed")


if __name__ == "__main__":
    main()