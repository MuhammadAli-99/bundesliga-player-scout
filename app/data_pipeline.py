import soccerdata as sd
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# All 7 competitions
LEAGUES = [
    "GER-Bundesliga",
    "ENG-Premier League",
    "ESP-La Liga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]
SEASON = "2324"

def fetch_player_stats(league: str, season: str) -> pd.DataFrame:
    """
    Fetch player standard stats (goals, assists, minutes, etc.) from FBref
    """
    fbref = sd.FBref(leagues=league, seasons=season)
    stats = fbref.read_player_season_stats(stat_type="standard")
    return stats

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten FBref's MultiIndex columns into single strings
    e.g. ('Performance', 'Gls') -> 'Performance_Gls'
    """
    df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in df.columns]
    return df.reset_index()

if __name__ == "__main__":
    all_data = []

    for league in LEAGUES:
        print(f"Fetching player stats for {league} {SEASON}...")
        try:
            df = fetch_player_stats(league, SEASON)
            df = flatten_columns(df)
            all_data.append(df)
            print(f"  -> {len(df)} players")
        except Exception as e:
            print(f"  -> Failed: {e}")

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal players across all leagues: {len(combined)}")
    print(f"\nColumns:\n{combined.columns.tolist()}")

    output_path = DATA_DIR / "all_leagues_player_stats_2324.csv"
    combined.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")