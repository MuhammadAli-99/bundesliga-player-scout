import soccerdata as sd
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

LEAGUES = [
    "GER-Bundesliga",
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]
SEASON = "2324"

if __name__ == "__main__":
    print("Testing SoFIFA player ratings...")
    sofifa = sd.SoFIFA(leagues=LEAGUES[0], versions="latest")

    print("\nFetching player ratings for Bundesliga...")
    ratings = sofifa.read_player_ratings()

    print(f"\nShape: {ratings.shape}")
    print(f"\nColumns:\n{ratings.columns.tolist()}")
    print(f"\nFirst 5 rows:\n{ratings.head()}")