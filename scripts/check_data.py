# scripts/check_data.py

import pandas as pd

matches = pd.read_csv("data/processed/matches_clean.csv")
deliveries = pd.read_csv("data/processed/deliveries_clean.csv")

print("\nMATCHES")
print(matches.columns.tolist())

print("\nDELIVERIES")
print(deliveries.columns.tolist())

print("\nMatches Shape:", matches.shape)
print("Deliveries Shape:", deliveries.shape)