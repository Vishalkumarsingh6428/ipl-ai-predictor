import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/processed")


def create_team_form(matches, window=5):
    matches = matches.copy()

    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values("date")

    records = []

    teams = pd.concat([
        matches["team1"],
        matches["team2"]
    ]).unique()

    for team in teams:

        team_matches = matches[
            (matches["team1"] == team) |
            (matches["team2"] == team)
        ].copy()

        team_matches = team_matches.sort_values("date")

        history = []

        for _, row in team_matches.iterrows():

            if len(history) == 0:
                form = 0.5
            else:
                form = sum(history[-window:]) / len(history[-window:])

            records.append({
                "match_id": row["id"],
                "team": team,
                "recent_win_rate": form
            })

            history.append(
                1 if row["winner"] == team else 0
            )

    return pd.DataFrame(records)


# Load cleaned data
matches = pd.read_csv(DATA_DIR / "matches_clean.csv")
deliveries = pd.read_csv(DATA_DIR / "deliveries_clean.csv")

# Keep only completed matches
matches = matches[matches["winner"].notna()]

# Create team form
team_form = create_team_form(matches)

# Merge match and delivery data
df = deliveries.merge(
    matches[
        [
            "id",
            "winner",
            "target_runs",
            "venue",
            "city",
            "toss_winner",
            "toss_decision"
        ]
    ],
    left_on="match_id",
    right_on="id",
    how="inner"
)

# Add batting team form
df = df.merge(
    team_form,
    left_on=["match_id", "batting_team"],
    right_on=["match_id", "team"],
    how="left"
)

df.rename(
    columns={"recent_win_rate": "batting_team_form"},
    inplace=True
)

df.drop(columns=["team"], inplace=True)

# Add bowling team form
df = df.merge(
    team_form,
    left_on=["match_id", "bowling_team"],
    right_on=["match_id", "team"],
    how="left"
)

df.rename(
    columns={"recent_win_rate": "bowling_team_form"},
    inplace=True
)

df.drop(columns=["team"], inplace=True)

# Only second innings
df = df[df["inning"] == 2].copy()

# Running score
df["current_score"] = (
    df.groupby("match_id")["total_runs"]
    .cumsum()
    .shift(1)
    .fillna(0)
)

# Runs left
df["runs_left"] = (
    df["target_runs"] - df["current_score"]
)

df = df[df["runs_left"] >= 0]

# Balls bowled
df["balls_bowled"] = (
    df["over"] * 6 + df["ball"]
)

df = df[df["balls_bowled"] > 0]

# Balls left
df["balls_left"] = 120 - df["balls_bowled"]

# Remove Super Over and miscounted deliveries
df = df[df["balls_left"] >= 0]

# Wickets fallen
df["wickets_fallen"] = (
    df.groupby("match_id")["is_wicket"]
    .cumsum()
)

# Wickets left
df["wickets_left"] = 10 - df["wickets_fallen"]

# Current Run Rate
df["crr"] = (
    df["current_score"] * 6
    / df["balls_bowled"]
)

# Required Run Rate
df["rrr"] = (
    df["runs_left"] * 6
    / df["balls_left"]
)

# Target variable
df["result"] = (
    df["batting_team"] == df["winner"]
).astype(int)

# Final feature set
features = df[
    [
        "match_id",
        "batting_team",
        "bowling_team",
        "venue",
        "city",
        "toss_winner",
        "toss_decision",
        "batting_team_form",
        "bowling_team_form",
        "runs_left",
        "balls_left",
        "wickets_left",
        "crr",
        "rrr",
        "result"
    ]
]

# Clean invalid values
features = features.replace(
    [float("inf"), -float("inf")],
    pd.NA
)

features = features.dropna()

# Save
features.to_csv(
    DATA_DIR / "match_features.csv",
    index=False
)

print("\nDataset Preview:")
print(features.head())

print("\nColumns:")
print(features.columns.tolist())

print("\nShape:", features.shape)

print("\nSaved to:")
print(DATA_DIR / "match_features.csv")