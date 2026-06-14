import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

df = pd.read_csv(DATA_DIR / "fifa_player_values.csv")

print(f"Shape: {df.shape}")
print(f"\nAll columns:\n{df.columns.tolist()}")

# Filter to our 5 leagues
our_leagues = ["Bundesliga", "Premier League", "La Liga", "Serie A", "Ligue 1"]
filtered = df[df["league_name"].isin(our_leagues)]

print(f"\nFiltered shape (our 5 leagues): {filtered.shape}")
print(f"\nPlayers per league:\n{filtered['league_name'].value_counts()}")

# Show key columns we care about
key_cols = ["short_name", "long_name", "club_name", "league_name", "value_eur", "wage_eur", "overall", "potential", "age"]
available_key_cols = [c for c in key_cols if c in filtered.columns]
print(f"\nKey columns available: {available_key_cols}")
print(f"\nSample data:\n{filtered[available_key_cols].head(10)}")