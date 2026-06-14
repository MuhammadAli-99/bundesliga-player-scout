import pandas as pd
from pathlib import Path
from unidecode import unidecode
from rapidfuzz import fuzz, process

DATA_DIR = Path("data")

LEAGUE_MAP = {
    "GER-Bundesliga": "Bundesliga",
    "ENG-Premier League": "Premier League",
    "ESP-La Liga": "La Liga",
    "ITA-Serie A": "Serie A",
    "FRA-Ligue 1": "Ligue 1",
}


def normalize_name(name: str) -> str:
    if pd.isna(name):
        return ""
    name = unidecode(str(name)).lower().strip()
    # strip any leftover non-ascii (e.g. CJK characters glued onto FIFA names)
    name = "".join(ch for ch in name if ch.isascii())
    return " ".join(name.split())


def load_fbref() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "all_leagues_full_stats_2324.csv")
    df["league_name"] = df["league"].map(LEAGUE_MAP)
    df["player_norm"] = df["player"].apply(normalize_name)
    df["team_norm"] = df["team"].apply(normalize_name)
    return df


def load_fifa() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "fifa_player_values.csv", low_memory=False)
    our_leagues = ["Bundesliga", "Premier League", "La Liga", "Serie A", "Ligue 1"]
    df = df[df["league_name"].isin(our_leagues)].copy()
    df["long_name_norm"] = df["long_name"].apply(normalize_name)
    df["short_name_norm"] = df["short_name"].apply(normalize_name)
    df["club_norm"] = df["club_name"].apply(normalize_name)
    return df


def load_fifa_all() -> pd.DataFrame:
    """Full FIFA dataset across all leagues — used for Stage 3 global search."""
    df = pd.read_csv(DATA_DIR / "fifa_player_values.csv", low_memory=False)
    df["long_name_norm"] = df["long_name"].apply(normalize_name)
    df["short_name_norm"] = df["short_name"].apply(normalize_name)
    df["club_norm"] = df["club_name"].apply(normalize_name)
    return df


def teams_match(team_norm: str, club_norm: str, threshold: int = 70) -> bool:
    if not team_norm or not club_norm:
        return False
    return fuzz.partial_ratio(team_norm, club_norm) >= threshold


def fuzzy_match_player(
    player_norm: str,
    team_norm: str,
    fifa_by_league: pd.DataFrame,
    fifa_all: pd.DataFrame = None,
    fbref_age: float = None,
):
    """
    Stage 1: team-restricted pool, token_set_ratio >= 80
    Stage 2: league-wide pool, token_set_ratio >= 96 (strict, avoids false positives)
    Stage 3: full FIFA dataset (all 43 leagues), token_set_ratio >= 85 + age within ±1
             (catches players who transferred outside our 5 competitions)
    Returns (fifa_index, score, stage) — stage=0 means no match.
    """
    team_mask = fifa_by_league["club_norm"].apply(lambda c: teams_match(team_norm, c))
    team_pool = fifa_by_league[team_mask]

    for pool, threshold, stage in [(team_pool, 80, 1), (fifa_by_league, 96, 2)]:
        if pool.empty:
            continue
        names_long = dict(zip(pool["long_name_norm"], pool.index))
        names_short = dict(zip(pool["short_name_norm"], pool.index))
        all_names = {**names_long, **names_short}
        all_names.pop("", None)
        if not all_names:
            continue
        result = process.extractOne(player_norm, all_names.keys(), scorer=fuzz.token_set_ratio)
        if result:
            best_match, score, _ = result
            if score >= threshold:
                return all_names[best_match], score, stage

    # Stage 3: global search with age corroboration
    if fifa_all is not None and fbref_age is not None and not pd.isna(fbref_age):
        names_long = dict(zip(fifa_all["long_name_norm"], fifa_all.index))
        names_short = dict(zip(fifa_all["short_name_norm"], fifa_all.index))
        all_names = {**names_long, **names_short}
        all_names.pop("", None)
        if all_names:
            result = process.extractOne(player_norm, all_names.keys(), scorer=fuzz.token_set_ratio)
            if result:
                best_match, score, _ = result
                if score >= 85:
                    idx = all_names[best_match]
                    fifa_age = fifa_all.loc[idx, "age"]
                    if not pd.isna(fifa_age) and abs(float(fifa_age) - float(fbref_age)) <= 1:
                        return idx, score, 3

    return None, 0, 0


if __name__ == "__main__":
    print("Loading FBref data...")
    fbref = load_fbref()
    print(f"  -> {len(fbref)} players")

    print("Loading FIFA data (5-league pool)...")
    fifa = load_fifa()
    print(f"  -> {len(fifa)} players")

    print("Loading FIFA data (full global pool for Stage 3)...")
    fifa_all = load_fifa_all()
    print(f"  -> {len(fifa_all)} players across all leagues")

    print("Matching players (3-stage)...")
    matched_indices = []
    match_scores = []
    match_stages = []

    for _, row in fbref.iterrows():
        league = row["league_name"]
        league_fifa = fifa[fifa["league_name"] == league]
        fifa_idx, score, stage = fuzzy_match_player(
            row["player_norm"], row["team_norm"], league_fifa,
            fifa_all=fifa_all, fbref_age=row["age"],
        )
        matched_indices.append(fifa_idx)
        match_scores.append(score)
        match_stages.append(stage)

    fbref["fifa_index"] = matched_indices
    fbref["match_score"] = match_scores
    fbref["match_stage"] = match_stages

    total = len(fbref)
    matched = fbref["fifa_index"].notna().sum()
    for s in [1, 2, 3]:
        n = (fbref["match_stage"] == s).sum()
        print(f"  Stage {s}: {n} players")
    print(f"\nMatched: {matched} / {total} ({matched/total*100:.1f}%)")

    # Merge FIFA columns — use full pool since Stage 3 matches come from outside our 5 leagues
    fifa_cols = ["value_eur", "wage_eur", "overall", "potential", "long_name", "club_name"]
    merged = fbref.merge(
        fifa_all[fifa_cols], left_on="fifa_index", right_index=True, how="left"
    )

    print(f"\nFinal merged shape: {merged.shape}")
    print(f"Players with market value: {merged['value_eur'].notna().sum()}")

    spot_check = merged[merged["player"].isin(
        ["Joshua Kimmich", "Iago", "Ritsu Doan", "Harry Kane", "Manuel Neuer"]
    )]
    print(f"\nSpot check:\n{spot_check[['player', 'long_name', 'team', 'club_name', 'value_eur', 'match_score', 'match_stage']].to_string()}")

    # Sample Stage 3 matches for false-positive sanity check
    stage3 = merged[merged["match_stage"] == 3][
        ["player", "long_name", "team", "club_name", "age", "value_eur", "match_score"]
    ].head(15)
    stage3_str = stage3.to_string().encode("ascii", errors="replace").decode("ascii")
    print(f"\n--- Stage 3 sample (age-verified cross-league matches) ---\n{stage3_str}")

    output_path = DATA_DIR / "merged_player_data.csv"
    merged.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")