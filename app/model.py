import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Performance-based features (the "honest" predictors)
PERFORMANCE_FEATURES = [
    "age", "age_squared", "years_to_peak",
    "Playing Time_90s", "Playing Time_Starts",
    "Performance_Gls", "Performance_Ast", "Performance_G+A",
    "Per 90 Minutes_Gls", "Per 90 Minutes_Ast",
    "shots_per90", "sot_per90", "Standard_SoT%",
    "crosses_per90", "interceptions_per90", "tackles_won_per90",
    "performance_index",
    "contract_years_remaining", "contract_urgent",
]

# FIFA rating features (variant B — risk of circularity/leakage)
FIFA_FEATURES = ["overall", "potential"]

# Wage feature (variant C — legitimate independent market signal)
WAGE_FEATURE = ["wage_eur"]

CATEGORICAL = ["pos_group", "league_name"]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "featured_player_data.csv")
    df = df[df["value_eur"].notna() & (df["reliable_sample"] == True)].copy()
    return df


def prepare_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Build the feature matrix: numeric features + one-hot encoded categoricals."""
    X = df[feature_cols].copy()
    cat_dummies = pd.get_dummies(df[CATEGORICAL], prefix=CATEGORICAL)
    X = pd.concat([X.reset_index(drop=True), cat_dummies.reset_index(drop=True)], axis=1)
    X = X.fillna(X.median(numeric_only=True))
    return X


def train_and_evaluate(df: pd.DataFrame, feature_cols: list, variant_name: str):
    """Train an XGBoost regressor and report MAE + R²."""
    X = prepare_features(df, feature_cols)
    # Predict log(value) — market values are highly right-skewed
    y = np.log1p(df["value_eur"].values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = np.expm1(model.predict(X_test))
    y_true = np.expm1(y_test)

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n=== {variant_name} ===")
    print(f"  Features: {X.shape[1]}")
    print(f"  Test set size: {len(y_test)}")
    print(f"  MAE:  EUR {mae:,.0f}")
    print(f"  R²:   {r2:.3f}")

    return model, X, mae, r2


def plot_feature_importance(model, X, variant_name, filename):
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({"feature": X.columns, "importance": importances})
    feat_imp = feat_imp.sort_values("importance", ascending=False).head(15)

    plt.figure(figsize=(10, 6))
    plt.barh(feat_imp["feature"][::-1], feat_imp["importance"][::-1], color="#667eea")
    plt.xlabel("Feature Importance")
    plt.title(f"Top 15 Features — {variant_name}")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / filename, dpi=120)
    plt.close()
    print(f"  Feature importance chart saved to models/{filename}")

    print(f"  Top 5 features:")
    for _, row in feat_imp.head(5).iterrows():
        print(f"    {row['feature']}: {row['importance']:.3f}")


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"  -> {len(df)} players with value + reliable sample")

    # Variant A — performance only (honest but limited by missing passing/possession stats)
    model_a, X_a, mae_a, r2_a = train_and_evaluate(
        df, PERFORMANCE_FEATURES, "Variant A: Performance-only"
    )
    plot_feature_importance(model_a, X_a, "Performance-only", "feature_importance_performance.png")

    # Variant B — performance + FIFA ratings (target leakage: overall ~ derived from value)
    model_b, X_b, mae_b, r2_b = train_and_evaluate(
        df, PERFORMANCE_FEATURES + FIFA_FEATURES, "Variant B: Performance + FIFA ratings"
    )
    plot_feature_importance(model_b, X_b, "Performance + FIFA", "feature_importance_fifa.png")

    # Variant C — performance + wage (legitimate independent market signal)
    model_c, X_c, mae_c, r2_c = train_and_evaluate(
        df, PERFORMANCE_FEATURES + WAGE_FEATURE, "Variant C: Performance + Wage"
    )
    plot_feature_importance(model_c, X_c, "Performance + Wage", "feature_importance_wage.png")

    # Comparison summary
    print("\n" + "=" * 55)
    print("COMPARISON")
    print("=" * 55)
    print(f"Variant A (performance-only):  MAE EUR {mae_a:>12,.0f} | R² {r2_a:.3f}")
    print(f"Variant B (+ FIFA ratings):    MAE EUR {mae_b:>12,.0f} | R² {r2_b:.3f}")
    print(f"Variant C (+ wage):            MAE EUR {mae_c:>12,.0f} | R² {r2_c:.3f}")
    print()
    print("Variant B's high R² is TARGET LEAKAGE — FIFA 'overall' is itself")
    print("derived from value_eur, so it is not a genuine predictor.")
    print("Variant C uses wage (an independent market signal) — the best")
    print("honest model, chosen as primary.")

    # Save Variant C as the primary model
    joblib.dump(model_c, MODEL_DIR / "market_value_model.joblib")
    joblib.dump(list(X_c.columns), MODEL_DIR / "model_features.joblib")
    print(f"\nPrimary model (Variant C) saved to models/market_value_model.joblib")