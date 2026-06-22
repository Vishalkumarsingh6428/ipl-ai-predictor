import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

MODELS_DIR = Path("models")
DATA_DIR   = Path("data/processed")

pipeline = joblib.load(MODELS_DIR / "xgboost.pkl")
df = pd.read_csv(DATA_DIR / "match_features.csv")
X  = df.drop(["result", "match_id"], axis=1)

preprocessor = pipeline.named_steps["preprocessor"]
model        = pipeline.named_steps["model"]

X_transformed = preprocessor.transform(X)

if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

print(f"Transformed shape: {X_transformed.shape}")

cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out([
    "batting_team", "bowling_team", "venue",
    "city", "toss_winner", "toss_decision"
])

numeric_feature_names = [
    "runs_left", "balls_left", "wickets_left",
    "crr", "rrr", "batting_team_form", "bowling_team_form"
]

all_feature_names = list(cat_feature_names) + numeric_feature_names
print(f"Total features: {len(all_feature_names)}")

X_df = pd.DataFrame(X_transformed, columns=all_feature_names)
sample = X_df.sample(1000, random_state=42)

print("Building SHAP explainer...")
explainer = shap.TreeExplainer(model)
joblib.dump(explainer, MODELS_DIR / "shap_explainer.pkl")
print("Saved: models/shap_explainer.pkl")

joblib.dump(all_feature_names, MODELS_DIR / "feature_names.pkl")
print("Saved: models/feature_names.pkl")

print("Generating SHAP summary plot...")
shap_values = explainer.shap_values(sample)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, sample, plot_type="bar", max_display=15, show=False)
plt.title("Top 15 Features by SHAP Importance")
plt.tight_layout()
plt.savefig(MODELS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: models/shap_summary.png")
