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

MERGE_KEYS = ["league", "season", "team", "player"]

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in df.columns]
    return df.reset_index()

def fetch_stat_type(league: str, season: str, stat_type: str) -> pd.DataFrame:
    fbref = sd.FBref(leagues=league, seasons=season)
    stats = fbref.read_player_season_stats(stat_type=stat_type)
    return flatten_columns(stats)

def merge_new_columns(base: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """
    Merge `new` into `base` on MERGE_KEYS, keeping only columns from `new`
    that don't already exist in `base`
    """
    new_cols = [c for c in new.columns if c in MERGE_KEYS or c not in base.columns]
    new = new[new_cols]
    return base.merge(new, on=MERGE_KEYS, how="left")

def fetch_all_stats_for_league(league: str, season: str) -> pd.DataFrame:
    print("  Fetching standard stats...")
    combined = fetch_stat_type(league, season, "standard")

    print("  Fetching shooting stats...")
    shooting = fetch_stat_type(league, season, "shooting")
    combined = merge_new_columns(combined, shooting)

    print("  Fetching misc stats...")
    misc = fetch_stat_type(league, season, "misc")
    combined = merge_new_columns(combined, misc)

    return combined

if __name__ == "__main__":
    all_data = []

    for league in LEAGUES:
        print(f"Fetching all stats for {league} {SEASON}...")
        try:
            df = fetch_all_stats_for_league(league, SEASON)
            all_data.append(df)
            print(f"  -> {len(df)} players, {len(df.columns)} columns")
        except Exception as e:
            print(f"  -> Failed: {e}")

    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal players: {len(combined)}")
    print(f"Total columns: {len(combined.columns)}")
    print(f"\nColumns:\n{combined.columns.tolist()}")

    output_path = DATA_DIR / "all_leagues_full_stats_2324.csv"
    combined.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")