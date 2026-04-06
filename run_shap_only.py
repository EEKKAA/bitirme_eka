"""
Standalone SHAP analysis script.
Loads the saved best model and dataset, then runs SHAP analysis.
Use this after fixing shap_analysis.py without re-running full training.
"""
import joblib
import json
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    OUTPUTS_DIR, DATASET_FILENAME, BEST_MODEL_FILENAME,
    SELECTED_RATIOS_FILENAME, TARGET,
)
from src.preprocessing import fit_winsorize_bounds, apply_winsorize_bounds
from src.shap_analysis import run_shap_analysis
from sklearn.preprocessing import MinMaxScaler


def main():
    # Load model
    print("Loading saved model...")
    best_model = joblib.load(BEST_MODEL_FILENAME)

    # Get display name from threshold config
    threshold_cfg = OUTPUTS_DIR / "threshold_config.json"
    if threshold_cfg.exists():
        with open(threshold_cfg) as f:
            best_name = json.load(f).get("model_name", type(best_model).__name__)
    else:
        best_name = type(best_model).__name__

    print(f"  Model: {best_name}  ({type(best_model).__name__})")

    # Load dataset first (we need it to determine feature set)
    print("Loading dataset...")
    df = pd.read_csv(DATASET_FILENAME)
    y = df[TARGET]

    # For StackingEnsemble: use the base model's own feature names if available
    # because each base model was trained on its own feature-selected subset.
    model_type = type(best_model).__name__
    if model_type == "StackingEnsemble":
        base_models = best_model.base_models
        shap_base = None
        for preferred in ("CatBoost", "LightGBM", "XGBoost"):
            if preferred in base_models:
                shap_base = base_models[preferred]
                break
        if shap_base is None:
            shap_base = next(iter(base_models.values()))

        # Get the feature list from the base model
        if hasattr(shap_base, "feature_names_") and shap_base.feature_names_:
            sel_features = list(shap_base.feature_names_)
        elif hasattr(shap_base, "feature_name_"):
            sel_features = list(shap_base.feature_name_())
        elif hasattr(shap_base, "get_booster"):
            sel_features = list(shap_base.get_booster().feature_names)
        else:
            # Fallback: load from saved ratios file
            with open(SELECTED_RATIOS_FILENAME, "r") as f:
                ratios_data = json.load(f)
            sel_features = ([item["feature"] for item in ratios_data["feature_ranking"]]
                            if isinstance(ratios_data, dict) else ratios_data)
    else:
        with open(SELECTED_RATIOS_FILENAME, "r") as f:
            ratios_data = json.load(f)
        if isinstance(ratios_data, list):
            sel_features = ratios_data
        elif isinstance(ratios_data, dict) and "feature_ranking" in ratios_data:
            sel_features = [item["feature"] for item in ratios_data["feature_ranking"]]
        else:
            sel_features = list(ratios_data.keys())

    print(f"  Selected features ({len(sel_features)}): {sel_features[:5]} ...")

    X = df[sel_features]

    # Apply same preprocessing (winsorize + scale)
    w_bounds = fit_winsorize_bounds(X, sel_features)
    X_w = apply_winsorize_bounds(X, w_bounds)
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_w),
        columns=sel_features,
        index=X_w.index,
    )

    # Run SHAP
    print("\nRunning SHAP analysis...")
    shap_result = run_shap_analysis(best_model, X_scaled, sel_features, model_name=best_name)

    if shap_result["feature_ranking"]:
        print("\n  Top Feature Ranking (by SHAP):")
        for item in shap_result["feature_ranking"][:10]:
            print(f"    {item['rank']:2d}. {item['feature']:40s} "
                  f"(|SHAP| = {item['mean_abs_shap']:.4f})")
    else:
        print("  SHAP did not produce a ranking.")

    print("\nDone.")


if __name__ == "__main__":
    main()
