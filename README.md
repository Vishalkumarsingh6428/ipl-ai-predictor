---
title: IPL AI Predictor 🏏
emoji: 🏏
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
---

# 🏏 IPL Win Probability Predictor

An end-to-end ML project that predicts **IPL match win probabilities** in real time using XGBoost and SHAP explainability.

## Features
- 🎯 Real-time win probability prediction for any match situation
- 🔍 SHAP explainability — see exactly which features drove each prediction
- 📊 Trained on 1,039 IPL matches (2008–2024) with 118k+ ball-by-ball records
- ✅ Leak-free evaluation using group-aware train/test split (match-level)

## Model Performance
| Metric | Score |
|--------|-------|
| Accuracy | 0.7836 |
| ROC-AUC | 0.8742 |
| Test matches | 208 (never seen during training) |

## Tech Stack
- **ML:** XGBoost, scikit-learn
- **Explainability:** SHAP
- **App:** Gradio
- **Data:** Kaggle IPL ball-by-ball dataset

## How It Works
1. Input the current match situation (teams, venue, score, overs, wickets)
2. The XGBoost model predicts win probability based on 160 engineered features
3. SHAP values explain which factors most influenced the prediction

## Project Structure
```
IPL/
├── scripts/
│   ├── preprocess.py          # Data cleaning
│   ├── feature_engineering.py # Feature creation (leak-free)
│   ├── train.py               # XGBoost training with GroupShuffleSplit
│   └── shap_explainer.py      # SHAP explainer generation
├── models/
│   ├── xgboost.pkl            # Trained model
│   ├── shap_explainer.pkl     # SHAP TreeExplainer
│   └── feature_names.pkl      # Feature name mapping
├── tests/
│   ├── test_features.py       # 11 feature validation tests
│   ├── test_model.py          # 9 model validation tests
│   └── test_preprocess.py     # 10 preprocessing tests
├── app.py                     # Gradio app
└── requirements.txt
```

## Author
**Vishal Kumar Singh** 
**IIT MADRAS**
