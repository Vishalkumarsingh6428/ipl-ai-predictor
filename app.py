import gradio as gr
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

pipeline       = joblib.load("models/xgboost.pkl")
explainer      = joblib.load("models/shap_explainer.pkl")
feature_names  = joblib.load("models/feature_names.pkl")

preprocessor = pipeline.named_steps["preprocessor"]
model        = pipeline.named_steps["model"]

TEAMS = sorted([
    "Chennai Super Kings", "Delhi Capitals", "Gujarat Titans",
    "Kolkata Knight Riders", "Lucknow Super Giants", "Mumbai Indians",
    "Punjab Kings", "Rajasthan Royals", "Royal Challengers Bangalore",
    "Sunrisers Hyderabad",
])

VENUES = sorted([
    "Wankhede Stadium", "M Chinnaswamy Stadium", "Eden Gardens",
    "MA Chidambaram Stadium", "Arun Jaitley Stadium",
    "Rajiv Gandhi International Cricket Stadium", "Sawai Mansingh Stadium",
    "Punjab Cricket Association IS Bindra Stadium",
    "Dr DY Patil Sports Academy", "Brabourne Stadium",
    "Narendra Modi Stadium",
])

CITIES = sorted([
    "Mumbai", "Bangalore", "Kolkata", "Chennai", "Delhi",
    "Hyderabad", "Jaipur", "Chandigarh", "Pune", "Ahmedabad",
])

def predict(batting_team, bowling_team, venue, city, toss_winner, toss_decision,
            target, current_score, overs_done, wickets_lost, batting_form, bowling_form):
    if batting_team == bowling_team:
        return "⚠️ Batting and bowling teams must be different.", None
    balls_bowled = int(overs_done * 6)
    balls_left   = 120 - balls_bowled
    runs_left    = target - current_score
    if balls_left <= 0:
        return "⚠️ No balls left — match is over.", None
    if runs_left < 0:
        return "✅ Batting team has already won!", None
    crr = (current_score * 6 / balls_bowled) if balls_bowled > 0 else 0.0
    rrr = (runs_left * 6 / balls_left)
    input_df = pd.DataFrame([{
        "batting_team": batting_team, "bowling_team": bowling_team,
        "venue": venue, "city": city, "toss_winner": toss_winner,
        "toss_decision": toss_decision, "batting_team_form": batting_form,
        "bowling_team_form": bowling_form, "runs_left": runs_left,
        "balls_left": balls_left, "wickets_left": 10 - wickets_lost,
        "crr": crr, "rrr": rrr,
    }])
    win_prob  = pipeline.predict_proba(input_df)[0][1]
    lose_prob = 1 - win_prob
    result_text = (
        f"🏏 **{batting_team}** win probability: **{win_prob*100:.1f}%**\n\n"
        f"🎯 **{bowling_team}** win probability: **{lose_prob*100:.1f}%**\n\n"
        f"📊 Required Run Rate: **{rrr:.2f}** | Current Run Rate: **{crr:.2f}**\n\n"
        f"🏃 Runs needed: **{runs_left}** off **{balls_left}** balls | "
        f"Wickets in hand: **{10 - wickets_lost}**"
    )
    X_transformed = preprocessor.transform(input_df)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    X_df      = pd.DataFrame(X_transformed, columns=feature_names)
    shap_vals = explainer.shap_values(X_df)
    vals      = shap_vals[0]
    top_idx   = np.argsort(np.abs(vals))[-10:][::-1]
    top_names = [feature_names[i] for i in top_idx]
    top_vals  = [vals[i] for i in top_idx]
    colors    = ["#2ecc71" if v > 0 else "#e74c3c" for v in top_vals]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(top_names)), [abs(v) for v in top_vals], color=colors, edgecolor="none")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("SHAP Value (impact on prediction)", fontsize=10)
    ax.set_title("Top 10 Features Driving This Prediction", fontsize=12, fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#2ecc71", label=f"Favours {batting_team}"),
        Patch(facecolor="#e74c3c", label=f"Favours {bowling_team}"),
    ], loc="lower right", fontsize=9)
    plt.tight_layout()
    return result_text, fig

with gr.Blocks(title="IPL AI Predictor 🏏", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏏 IPL Win Probability Predictor\n**XGBoost | ROC-AUC: 0.8742 | SHAP Explainability**")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Match Setup")
            batting_team  = gr.Dropdown(TEAMS, label="Batting Team", value="Mumbai Indians")
            bowling_team  = gr.Dropdown(TEAMS, label="Bowling Team", value="Chennai Super Kings")
            venue         = gr.Dropdown(VENUES, label="Venue", value="Wankhede Stadium")
            city          = gr.Dropdown(CITIES, label="City", value="Mumbai")
            toss_winner   = gr.Dropdown(TEAMS, label="Toss Winner", value="Mumbai Indians")
            toss_decision = gr.Radio(["bat", "field"], label="Toss Decision", value="field")
        with gr.Column():
            gr.Markdown("### Match Situation")
            target        = gr.Number(label="Target", value=180, minimum=1)
            current_score = gr.Number(label="Current Score", value=80, minimum=0)
            overs_done    = gr.Slider(0, 20, value=10, step=0.1, label="Overs Completed")
            wickets_lost  = gr.Slider(0, 9, value=2, step=1, label="Wickets Lost")
            batting_form  = gr.Slider(0, 1, value=0.6, step=0.1, label="Batting Team Form (0-1)")
            bowling_form  = gr.Slider(0, 1, value=0.5, step=0.1, label="Bowling Team Form (0-1)")
    predict_btn = gr.Button("🔮 Predict", variant="primary", size="lg")
    with gr.Row():
        result_text = gr.Markdown()
        shap_plot   = gr.Plot(label="SHAP Feature Importance")
    predict_btn.click(
        fn=predict,
        inputs=[batting_team, bowling_team, venue, city, toss_winner, toss_decision,
                target, current_score, overs_done, wickets_lost, batting_form, bowling_form],
        outputs=[result_text, shap_plot],
    )

if __name__ == "__main__":
    demo.launch()
