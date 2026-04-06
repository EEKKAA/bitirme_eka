"""
SHAP explainability module for the best-performing distress prediction model.
Generates global importance, beeswarm plot, and feature ranking.
"""
import numpy as np
import pandas as pd
import json

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- GLOBAL STYLING TO MATCH HTML PRESENTATION ---
TEXT_MAIN = '#f8fafc'
TEXT_MUTED = '#cbd5e1'
ACCENT_GREEN = '#10b981'
ACCENT_RED = '#ef4444'
ACCENT_BLUE = '#3b82f6'

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': 'none',
    'axes.facecolor': 'none',
    'savefig.facecolor': 'none',
    'axes.edgecolor': TEXT_MUTED,
    'axes.labelcolor': TEXT_MAIN,
    'xtick.color': TEXT_MUTED,
    'ytick.color': TEXT_MUTED,
    'text.color': TEXT_MAIN,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Arial', 'Inter', 'sans-serif'],
    'axes.grid': False,
    'grid.alpha': 0.1,
    'grid.color': '#ffffff'
})
# --- END GLOBAL STYLING ---

from config import PLOTS_DIR, SELECTED_RATIOS_FILENAME


def run_shap_analysis(model, X: pd.DataFrame, feature_names: list,
                      model_name: str = "Best Model") -> dict:
    """
    Runs SHAP analysis on the given model and generates:
      1. Global feature importance bar chart
      2. Beeswarm plot
      3. Feature ranking saved to JSON

    Args:
        model: Trained model (must be tree-based or support SHAP).
        X: Feature matrix used for SHAP values (scaled).
        feature_names: List of feature names.
        model_name: Display name for plots.

    Returns:
        Dict with 'feature_ranking' (sorted list of {name, importance}).
    """
    if not SHAP_AVAILABLE:
        print("Warning: SHAP not installed. Skipping SHAP analysis.")
        return {"feature_ranking": []}

    print(f"  Computing SHAP values for {model_name}...")

    # Create appropriate explainer
    model_type = type(model).__name__
    shap_model_label = model_name   # used in plot titles

    if model_type == "StackingEnsemble":
        # StackingEnsemble is a custom class — SHAP cannot introspect it directly.
        # Extract the best tree base model (preference: CatBoost > LightGBM > XGBoost)
        # and run TreeExplainer on it.  Results represent the dominant base learner,
        # which is very close to the ensemble in predictive power (AUC diff < 0.01).
        base_models = model.base_models  # {name: fitted_estimator}
        shap_base = None
        for preferred in ("CatBoost", "LightGBM", "XGBoost"):
            if preferred in base_models:
                shap_base = base_models[preferred]
                shap_model_label = f"{model_name} — {preferred} base"
                print(f"  StackingEnsemble detected: using {preferred} base model for SHAP.")
                break
        if shap_base is None:
            # Fallback: use first available base model
            first_name, shap_base = next(iter(base_models.items()))
            shap_model_label = f"{model_name} — {first_name} base"

        # Reorder X to match the base model's training feature order.
        # CatBoost/LightGBM/XGBoost store feature names and require the same column order.
        base_feat_names = None
        if hasattr(shap_base, "feature_names_"):
            base_feat_names = list(shap_base.feature_names_)
        elif hasattr(shap_base, "feature_name_"):   # LightGBM
            base_feat_names = list(shap_base.feature_name_())
        elif hasattr(shap_base, "get_booster"):     # XGBoost
            base_feat_names = list(shap_base.get_booster().feature_names)
        if base_feat_names and set(base_feat_names) <= set(X.columns):
            X = X[base_feat_names]
            feature_names = base_feat_names

        explainer = shap.TreeExplainer(shap_base)
    elif model_type in ("XGBClassifier", "CatBoostClassifier",
                        "RandomForestClassifier", "GradientBoostingClassifier",
                        "LGBMClassifier"):
        explainer = shap.TreeExplainer(model)
    else:
        # Logistic Regression or other linear models
        explainer = shap.LinearExplainer(model, X)

    shap_values = explainer.shap_values(X)

    # Handle multi-output SHAP values (some models return list)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # class 1 = distress

    # ── 1. Global Feature Importance Bar Chart ────────────────────────────
    print("  Generating SHAP global importance plot...")
    plt.figure(figsize=(10, 6))
    plt.gcf().patch.set_facecolor('none')
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=20, color=ACCENT_BLUE)
    plt.title(f"SHAP Global Feature Importance — {shap_model_label}", fontsize=18, fontweight="bold", color=TEXT_MAIN, pad=20)
    plt.xlabel("Mean |SHAP Value| (Impact on Model Output)", fontsize=14, fontweight="bold", color=TEXT_MAIN)
    plt.xticks(fontsize=12, color=TEXT_MUTED)
    plt.yticks(fontsize=12, color=TEXT_MAIN)
    plt.grid(axis='x', alpha=0.1, color='#ffffff')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_global_importance.png", dpi=300,
                bbox_inches="tight", facecolor='none')
    plt.close()

    # ── 2. Beeswarm Plot ──────────────────────────────────────────────────
    print("  Generating SHAP beeswarm plot...")
    plt.figure(figsize=(10, 8))
    plt.gcf().patch.set_facecolor('none')
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                      show=False, max_display=20)
    plt.title(f"SHAP Beeswarm — {shap_model_label}", fontsize=18, fontweight="bold", color=TEXT_MAIN, pad=20)
    plt.xlabel("SHAP Value (Decision Impact)", fontsize=14, fontweight="bold", color=TEXT_MAIN)
    plt.xticks(fontsize=12, color=TEXT_MUTED)
    plt.yticks(fontsize=12, color=TEXT_MAIN)
    plt.grid(alpha=0.1, color='#ffffff')
    
    cb = plt.gcf().axes[-1] 
    if cb: cb.set_ylabel('Feature Value', size=14, color=TEXT_MAIN)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_beeswarm.png", dpi=300,
                bbox_inches="tight", facecolor='none')
    plt.close()

    # ── 3. Feature Ranking ────────────────────────────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    ranking = sorted(
        zip(feature_names, mean_abs_shap),
        key=lambda x: x[1], reverse=True
    )
    feature_ranking = [
        {"rank": i + 1, "feature": name, "mean_abs_shap": float(score)}
        for i, (name, score) in enumerate(ranking)
    ]

    # Save ranking to JSON
    with open(SELECTED_RATIOS_FILENAME, "w") as f:
        json.dump({
            "model": model_name,
            "feature_ranking": feature_ranking
        }, f, indent=2)
    print(f"  Feature ranking saved to {SELECTED_RATIOS_FILENAME}")

    return {"feature_ranking": feature_ranking}
