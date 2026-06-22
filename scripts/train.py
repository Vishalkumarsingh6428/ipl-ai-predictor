import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Load data
df = pd.read_csv("data/processed/match_features.csv")

print("Total rows:", len(df))
print("Unique matches:", df["match_id"].nunique())

# Separate group key
groups = df["match_id"]

X = df.drop(["result", "match_id"], axis=1)
y = df["result"]

# Features
categorical_features = [
    "batting_team",
    "bowling_team",
    "venue",
    "city",
    "toss_winner",
    "toss_decision"
]

numeric_features = [
    "runs_left",
    "balls_left",
    "wickets_left",
    "crr",
    "rrr",
    "batting_team_form",
    "bowling_team_form"
]

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            StandardScaler(),
            numeric_features
        )
    ]
)

# Model
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

# Pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Group-aware split — ensures entire matches stay in train or test only
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups))

X_train = X.iloc[train_idx]
X_test  = X.iloc[test_idx]
y_train = y.iloc[train_idx]
y_test  = y.iloc[test_idx]

print("\nTrain rows:", len(X_train))
print("Test rows: ", len(X_test))

# Train
pipeline.fit(X_train, y_train)

# Evaluate
preds = pipeline.predict(X_test)
probs = pipeline.predict_proba(X_test)[:, 1]

print("\nAccuracy:", round(accuracy_score(y_test, preds), 4))
print("ROC-AUC: ", round(roc_auc_score(y_test, probs), 4))

# Save
joblib.dump(pipeline, "models/xgboost.pkl")
print("\nModel saved to models/xgboost.pkl")