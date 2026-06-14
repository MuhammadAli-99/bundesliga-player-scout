import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
CURRENT_YEAR = 2026
MIN_90S_THRESHOLD = 10  # ~900 minutes, roughly a third of a season


def load_merged() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "merged_player_data.csv")


def load_fifa_contract_info() -> pd.DataFrame:
    """
    Re-load the FIFA dataset and keep only the contract column,
    indexed the same way as during merge_data.py (so fifa_index aligns).
    """
    df = pd.read_csv(DATA_DIR / "fifa_player_values.csv", low_memory=False)
    return df[["club_contract_valid_until_year"]]


def add_per90_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create per-90 metrics for stats that don't already have them.
    Avoid division by zero using a small floor on minutes played.
    """
    minutes_90s = df["Playing Time_90s"].replace(0, np.nan)

    df["shots_per90"] = df["Standard_Sh"] / minutes_90s
    df["sot_per90"] = df["Standard_SoT"] / minutes_90s
    df["crosses_per90"] = df["Performance_Crs"] / minutes_90s
    df["interceptions_per90"] = df["Performance_Int"] / minutes_90s
    df["tackles_won_per90"] = df["Performance_TklW"] / minutes_90s
    df["fouls_per90"] = df["Performance_Fls"] / minutes_90s

    per90_cols = ["shots_per90", "sot_per90", "crosses_per90",
                   "interceptions_per90", "tackles_won_per90", "fouls_per90"]
    df[per90_cols] = df[per90_cols].fillna(0)

    return df


def get_position_group(pos: str) -> str:
    """
    Simplify FBref's multi-position strings (e.g. 'MF,FW') into
    a single primary group: GK, DF, MF, FW
    """
    if pd.isna(pos):
        return "MF"
    pos = str(pos)
    if "GK" in pos:
        return "GK"
    if "DF" in pos:
        return "DF"
    if "FW" in pos:
        return "FW"
    return "MF"


def add_performance_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite performance score (0-100 scale, roughly) combining
    attacking and defensive contributions, weighted by position group.

    Attacking score: goals/90 + assists/90 + shots on target/90 (scaled)
    Defensive score: tackles won/90 + interceptions/90 (scaled)

    FW/MF -> attacking weighted higher
    DF/GK -> defensive weighted higher
    """
    df["pos_group"] = df["pos"].apply(get_position_group)

    attacking_score = (
        df["Per 90 Minutes_Gls"].fillna(0) * 10
        + df["Per 90 Minutes_Ast"].fillna(0) * 8
        + df["sot_per90"] * 3
    )

    defensive_score = (
        df["tackles_won_per90"] * 4
        + df["interceptions_per90"] * 4
    )

    weights = {
        "FW": (0.8, 0.2),
        "MF": (0.5, 0.5),
        "DF": (0.2, 0.8),
        "GK": (0.1, 0.9),
    }

    att_w = df["pos_group"].map(lambda p: weights[p][0])
    def_w = df["pos_group"].map(lambda p: weights[p][1])

    df["performance_index"] = (attacking_score * att_w + defensive_score * def_w).round(2)

    return df


def flag_reliable_sample(df: pd.DataFrame, min_90s: float = MIN_90S_THRESHOLD) -> pd.DataFrame:
    """
    Flag players with enough playing time for per-90 stats to be meaningful.
    Players below the threshold get NaN for performance_index and per-90
    metrics, rather than misleadingly extreme values from tiny denominators
    (e.g. 1 shot in 9 minutes -> 10 shots/90).
    """
    df["reliable_sample"] = df["Playing Time_90s"] >= min_90s

    per90_cols = [
        "shots_per90", "sot_per90", "crosses_per90",
        "interceptions_per90", "tackles_won_per90", "fouls_per90",
        "Per 90 Minutes_Gls", "Per 90 Minutes_Ast", "Per 90 Minutes_G+A",
        "performance_index"
    ]

    for col in per90_cols:
        df.loc[~df["reliable_sample"], col] = np.nan

    return df


def add_age_value_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quadratic age feature captures the typical rise-then-decline
    pattern of player market value with age. Peak age assumed ~27.
    """
    PEAK_AGE = 27
    df["age_squared"] = df["age"] ** 2
    df["age_diff_from_peak"] = (df["age"] - PEAK_AGE).abs()
    df["years_to_peak"] = PEAK_AGE - df["age"]
    return df


def add_contract_urgency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge in contract expiry year using the saved fifa_index, then
    compute years remaining and an urgency flag.
    """
    fifa_contracts = load_fifa_contract_info()

    df["fifa_index"] = pd.to_numeric(df["fifa_index"], errors="coerce")
    df = df.merge(
        fifa_contracts, left_on="fifa_index", right_index=True, how="left"
    )

    df["contract_years_remaining"] = df["club_contract_valid_until_year"] - CURRENT_YEAR
    df["contract_urgent"] = (df["contract_years_remaining"] <= 1).astype(int)

    return df


if __name__ == "__main__":
    print("Loading merged data...")
    df = load_merged()
    print(f"  -> {df.shape}")

    print("Adding per-90 metrics...")
    df = add_per90_metrics(df)

    print("Adding performance index...")
    df = add_performance_index(df)

    print(f"Flagging reliable samples (>= {MIN_90S_THRESHOLD} x 90-minute equivalents)...")
    df = flag_reliable_sample(df)

    print("Adding age-value curve features...")
    df = add_age_value_features(df)

    print("Adding contract urgency...")
    df = add_contract_urgency(df)

    print(f"\nFinal shape: {df.shape}")

    new_cols = [
        "player", "team", "pos_group", "age", "value_eur",
        "performance_index", "shots_per90", "sot_per90",
        "tackles_won_per90", "interceptions_per90",
        "age_squared", "years_to_peak",
        "contract_years_remaining", "contract_urgent"
    ]

    print(f"\nSample (top performers by performance_index, reliable sample only):")
    print(df[df["reliable_sample"]].dropna(subset=["value_eur", "performance_index"]).sort_values("performance_index", ascending=False)[new_cols].head(10))

    print(f"\nReliable sample distribution:")
    print(df["reliable_sample"].value_counts())

    print(f"\nContract urgency distribution:")
    print(df["contract_urgent"].value_counts())

    output_path = DATA_DIR / "featured_player_data.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")