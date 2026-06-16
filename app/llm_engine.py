import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path("C:/Users/HP/bundesliga-player-scout/.env"))

import pandas as pd
import numpy as np
import joblib
import anthropic

DATA_DIR = Path("data")
MODEL_DIR = Path("models")

# Reuse the exact feature-prep logic from the model
from app.model import prepare_features, PERFORMANCE_FEATURES, WAGE_FEATURE


def get_client():
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def load_model_and_data():
    model = joblib.load(MODEL_DIR / "market_value_model.joblib")
    feature_order = joblib.load(MODEL_DIR / "model_features.joblib")
    df = pd.read_csv(DATA_DIR / "featured_player_data.csv")
    df = df[df["value_eur"].notna() & (df["reliable_sample"] == True)].copy()
    return model, feature_order, df


def predict_value(model, feature_order, df) -> pd.DataFrame:
    """Add a predicted_value_eur column to the dataframe."""
    X = prepare_features(df, PERFORMANCE_FEATURES + WAGE_FEATURE)
    X = X.reindex(columns=feature_order, fill_value=0)
    df = df.copy()
    df["predicted_value_eur"] = np.expm1(model.predict(X))
    df["value_gap_pct"] = ((df["predicted_value_eur"] - df["value_eur"]) / df["value_eur"] * 100).round(1)
    return df


def get_verdict(value_gap_pct: float) -> str:
    """Classify based on how far predicted value is from actual."""
    if value_gap_pct >= 20:
        return "UNDERVALUED"
    elif value_gap_pct <= -20:
        return "OVERVALUED"
    else:
        return "FAIRLY PRICED"


def build_player_summary(row: pd.Series) -> str:
    """Build a compact, readable stat summary for the LLM prompt."""
    return f"""
Player: {row['player']}
Club: {row['team']} ({row['league_name']})
Position: {row['pos_group']}
Age: {int(row['age'])}
Minutes played (90s): {row['Playing Time_90s']:.1f}
Goals: {int(row['Performance_Gls'])}, Assists: {int(row['Performance_Ast'])}
Goals per 90: {row['Per 90 Minutes_Gls']:.2f}, Assists per 90: {row['Per 90 Minutes_Ast']:.2f}
Shots on target per 90: {row['sot_per90']:.2f}
Tackles won per 90: {row['tackles_won_per90']:.2f}, Interceptions per 90: {row['interceptions_per90']:.2f}
Performance index: {row['performance_index']:.1f}
Contract years remaining: {row['contract_years_remaining']:.0f}

Actual market value: EUR {row['value_eur']:,.0f}
Model-predicted value: EUR {row['predicted_value_eur']:,.0f}
Value gap: {row['value_gap_pct']:+.1f}% ({get_verdict(row['value_gap_pct'])})
""".strip()


def generate_scouting_report(row: pd.Series, language: str = "en") -> str:
    """Generate a ~250-word professional scouting report via Claude."""
    client = get_client()
    summary = build_player_summary(row)
    lang_instruction = "Write the report in German." if language == "de" else "Write the report in English."

    prompt = f"""You are a professional football scout and data analyst preparing a transfer recommendation for a club's recruitment department.

Based on the player data below, write a concise scouting report of approximately 250 words.

{summary}

{lang_instruction}

Structure the report as follows:
- Opening verdict: state clearly whether the player appears UNDERVALUED, OVERVALUED, or FAIRLY PRICED relative to the model's prediction, and by how much.
- Performance analysis: interpret their on-pitch output (goals, assists, defensive actions) for their position. Be specific with the numbers.
- Value context: explain what's driving the gap between actual and predicted value (e.g. contract situation, age, output level, league).
- Recommendation: a clear, actionable closing line for the recruitment team.

Keep the tone professional and analytical, suitable for a club's data-driven scouting department. Do not invent statistics not provided. Note that the predicted value comes from a model trained on performance and wage data, so it may not capture reputation or playmaking quality fully."""

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


if __name__ == "__main__":
    print("Loading model and data...")
    model, feature_order, df = load_model_and_data()
    df = predict_value(model, feature_order, df)

    # Test on a few well-known players
    test_players = ["Harry Kane", "Florian Wirtz", "Erling Haaland"]

    for name in test_players:
        match = df[df["player"] == name]
        if match.empty:
            print(f"\n{name} not found in dataset.")
            continue
        row = match.iloc[0]
        print("\n" + "=" * 60)
        print(f"SCOUTING REPORT: {name}")
        print("=" * 60)
        print(generate_scouting_report(row, language="en"))