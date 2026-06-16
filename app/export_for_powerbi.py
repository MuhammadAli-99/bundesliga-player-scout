import pandas as pd
from pathlib import Path
from app.llm_engine import load_model_and_data, predict_value, get_verdict

DATA_DIR = Path("data")

if __name__ == "__main__":
    print("Loading model and computing predictions...")
    model, feature_order, df = load_model_and_data()
    df = predict_value(model, feature_order, df)

    # Data quality filters for dashboard credibility:
    # 1. Exclude near-zero market values (cause divide-by-zero in value gap,
    #    e.g. retired/veteran players listed at 0)
    # 2. Exclude goalkeepers — model has no GK-specific features (saves, clean
    #    sheets), so GK valuations are unreliable and out of scope
    before = len(df)
    df = df[df["value_eur"] >= 250000]
    df = df[df["pos_group"] != "GK"]
    print(f"Filtered {before - len(df)} rows (near-zero values + goalkeepers); {len(df)} remain")

    df["verdict"] = df["value_gap_pct"].apply(get_verdict)

    # Select and rename columns for a clean Power BI dataset
    export_cols = {
        "player": "Player",
        "team": "Club",
        "league_name": "League",
        "pos_group": "Position",
        "age": "Age",
        "nation": "Nationality",
        "Playing Time_90s": "Matches_90s",
        "Performance_Gls": "Goals",
        "Performance_Ast": "Assists",
        "Per 90 Minutes_Gls": "Goals_per90",
        "Per 90 Minutes_Ast": "Assists_per90",
        "sot_per90": "ShotsOnTarget_per90",
        "tackles_won_per90": "TacklesWon_per90",
        "interceptions_per90": "Interceptions_per90",
        "performance_index": "PerformanceIndex",
        "value_eur": "MarketValue_EUR",
        "predicted_value_eur": "PredictedValue_EUR",
        "value_gap_pct": "ValueGap_Pct",
        "verdict": "Verdict",
        "wage_eur": "Wage_EUR",
        "overall": "FIFA_Overall",
        "potential": "FIFA_Potential",
        "contract_years_remaining": "ContractYearsLeft",
        "contract_urgent": "ContractUrgent",
    }

    export_df = df[list(export_cols.keys())].rename(columns=export_cols)

    # Round numeric columns for cleaner display
    export_df["PredictedValue_EUR"] = export_df["PredictedValue_EUR"].round(0)
    export_df["Goals_per90"] = export_df["Goals_per90"].round(2)
    export_df["Assists_per90"] = export_df["Assists_per90"].round(2)
    export_df["ShotsOnTarget_per90"] = export_df["ShotsOnTarget_per90"].round(2)
    export_df["TacklesWon_per90"] = export_df["TacklesWon_per90"].round(2)
    export_df["Interceptions_per90"] = export_df["Interceptions_per90"].round(2)

    output_path = DATA_DIR / "powerbi_dataset.csv"
    export_df.to_csv(output_path, index=False)
    print(f"Saved {len(export_df)} players to {output_path}")
    print(f"\nColumns: {export_df.columns.tolist()}")
    print(f"\nVerdict distribution:\n{export_df['Verdict'].value_counts()}")
    print(f"\nSample (top undervalued):")
    print(export_df[export_df['Verdict'] == 'UNDERVALUED'].sort_values('ValueGap_Pct', ascending=False)[['Player', 'Club', 'MarketValue_EUR', 'PredictedValue_EUR', 'ValueGap_Pct']].head(10).to_string())