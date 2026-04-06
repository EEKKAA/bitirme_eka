"""
KMV vs ML Models — Comparison Pipeline

Runs the KMV (Merton) structural model alongside all ML models
and generates stakeholder-ready comparison visualizations:
  1. AUC Bar Chart with KMV benchmark line
  2. ROC Curve Overlay (all models + KMV on one plot)
  3. Radar Chart (multi-metric comparison)
  4. Summary comparison table
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap

# --- GLOBAL STYLING TO MATCH HTML PRESENTATION ---
BG_SURFACE = '#1e293b'
TEXT_MAIN = '#f8fafc'
TEXT_MUTED = '#cbd5e1'
ACCENT_GREEN = '#10b981'
ACCENT_RED = '#ef4444'
ACCENT_BLUE = '#3b82f6'
ACCENT_PURPLE = '#a855f7' # Added for CatBoost

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

from config import (
    DATASET_FILENAME, CV_RESULTS_FILENAME, PLOTS_DIR, OUTPUTS_DIR,
    CANDIDATE_FEATURES, TARGET, RANDOM_STATE, FEATURE_SELECTION_K,
)
from src.preprocessing import preprocess_data
from src.kmv_model import run_kmv_model


def load_cv_results():
    """Load cross-validation results from the last training run."""
    if not CV_RESULTS_FILENAME.exists():
        print("Error: CV results not found. Run train_models.py first.")
        return None
    return pd.read_csv(CV_RESULTS_FILENAME)


def plot_auc_comparison(ml_results: dict, kmv_metrics: dict, filename: str = "kmv_vs_ml_auc.png"):
    """
    Bar chart comparing AUC across all models with KMV benchmark.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('none') # Make figure background transparent

    # Prepare data
    models = list(ml_results.keys()) + ["KMV (Merton)"]
    aucs = []
    for m in ml_results:
        v = ml_results[m]
        aucs.append(v["auc"] if isinstance(v, dict) else v)
    aucs.append(kmv_metrics.get("auc", 0))
    colors = []
    for m in models:
        if "KMV" in m:
            colors.append(ACCENT_RED)
        elif "XGBoost" in m:
            colors.append(ACCENT_GREEN)
        elif "LightGBM" in m:
            colors.append(ACCENT_BLUE)
        elif "CatBoost" in m:
            colors.append(ACCENT_PURPLE)
        else:
            colors.append(TEXT_MUTED)

    bars = ax.barh(models, aucs, color=colors, edgecolor="white", height=0.6)

    # Add value labels
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f"{auc:.4f}", va="center", fontsize=12, fontweight="bold", color=TEXT_MAIN)

    # KMV benchmark line
    kmv_auc = kmv_metrics.get("auc", 0)
    ax.axvline(x=kmv_auc, color=ACCENT_RED, linestyle="--", linewidth=2, alpha=0.7,
               label=f"KMV Benchmark (AUC={kmv_auc:.4f})")

    ax.set_xlabel("AUC Score", fontsize=13, fontweight="bold", color=TEXT_MAIN)
    ax.set_title("Model Comparison: ML Models vs KMV (Merton)\nBIST Financial Distress Prediction",
                 fontsize=15, fontweight="bold", pad=15, color=TEXT_MAIN)
    ax.set_xlim(0, 1.05)
    legend = ax.legend(loc="lower right", fontsize=11)
    legend.get_frame().set_facecolor(BG_SURFACE)
    legend.get_frame().set_edgecolor(TEXT_MUTED)
    for text in legend.get_texts(): text.set_color(TEXT_MAIN)
    ax.grid(axis="x", alpha=0.1, color='#ffffff') # Use global grid style

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight", facecolor='none')
    plt.close()
    print(f"  [OK] AUC comparison chart -> {filename}")


def plot_roc_overlay(ml_roc_data: dict, kmv_roc: dict, filename: str = "kmv_vs_ml_roc.png"):
    """
    Overlaid ROC curves for all models including KMV.
    """
    fig, ax = plt.subplots(figsize=(10, 9))
    fig.patch.set_facecolor('none') # Make figure background transparent

    color_map = {
        "XGBoost (No SMOTE)": ACCENT_GREEN,
        "CatBoost (No SMOTE)": ACCENT_PURPLE,
        "LightGBM (No SMOTE)": ACCENT_BLUE,
        "Logistic Regression (No SMOTE)": TEXT_MUTED,
        "KMV (Merton)": ACCENT_RED,
    }

    # Plot ML model ROC curves
    for model_name, roc_info in ml_roc_data.items():
        color = color_map.get(model_name, TEXT_MUTED)
        ax.plot(roc_info["fpr"], roc_info["tpr"],
                label=f"{model_name} (AUC={roc_info['auc']:.4f})",
                linewidth=2.5, color=color)

    # Plot KMV ROC curve
    if kmv_roc and "fpr" in kmv_roc:
        ax.plot(kmv_roc["fpr"], kmv_roc["tpr"],
                label=f"KMV Merton (AUC={kmv_roc['auc']:.4f})",
                linewidth=3, color=ACCENT_RED, linestyle="--")

    # Diagonal
    ax.plot([0, 1], [0, 1], color=TEXT_MUTED, linestyle=":", linewidth=1)

    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=13, color=TEXT_MAIN)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=13, color=TEXT_MAIN)
    ax.set_title("ROC Curve Comparison: ML Models vs KMV (Merton)\nBIST Financial Distress Prediction",
                 fontsize=14, fontweight="bold", pad=15, color=TEXT_MAIN)
    legend = ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    legend.get_frame().set_facecolor(BG_SURFACE)
    legend.get_frame().set_edgecolor(TEXT_MUTED)
    for text in legend.get_texts(): text.set_color(TEXT_MAIN)
    ax.grid(alpha=0.1, color='#ffffff') # Use global grid style
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] ROC overlay chart -> {filename}")


def plot_radar_comparison(all_metrics: dict, filename: str = "kmv_vs_ml_radar.png"):
    """
    Radar/spider chart comparing multiple metrics across models.
    """
    categories = ["AUC", "F1", "Precision", "Recall", "MCC", "Accuracy"]
    metric_keys = ["auc", "f1", "precision", "recall", "mcc", "accuracy"]
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    color_map = {
        "XGBoost (No SMOTE)": "#2ECC71",
        "CatBoost (No SMOTE)": "#9B59B6",
        "LightGBM (No SMOTE)": "#3498DB",
        "Logistic Regression (No SMOTE)": "#95A5A6",
        "KMV (Merton)": "#E74C3C",
    }

    for model_name, metrics in all_metrics.items():
        values = [metrics.get(k, 0) for k in metric_keys]
        # Normalize MCC from [-1,1] to [0,1] for radar
        mcc_idx = metric_keys.index("mcc")
        values[mcc_idx] = (values[mcc_idx] + 1) / 2
        values += values[:1]

        color = color_map.get(model_name, "#34495E")
        linestyle = "--" if "KMV" in model_name else "-"
        linewidth = 3 if "KMV" in model_name else 2

        ax.plot(angles, values, "o-", linewidth=linewidth, label=model_name,
                color=color, linestyle=linestyle)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title("Multi-Metric Comparison: ML vs KMV\n(MCC normalized to 0-1 scale)",
                 fontsize=14, fontweight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Radar comparison chart -> {filename}")


def plot_summary_table(all_metrics: dict, filename: str = "kmv_vs_ml_table.png"):
    """
    Visual summary table of all model metrics.
    """
    metrics_display = ["AUC", "F1", "Precision", "Recall", "MCC", "Accuracy"]
    metric_keys = ["auc", "f1", "precision", "recall", "mcc", "accuracy"]

    models = list(all_metrics.keys())
    data = []
    for model in models:
        row = [f"{all_metrics[model].get(k, 0):.4f}" for k in metric_keys]
        data.append(row)

    fig, ax = plt.subplots(figsize=(16, len(models) * 0.8 + 2.5))
    ax.axis("off")

    table = ax.table(
        cellText=data,
        rowLabels=models,
        colLabels=metrics_display,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.0, 2.0)

    # Style header
    for j in range(len(metrics_display)):
        cell = table[0, j]
        cell.set_facecolor(BG_SURFACE)
        cell.set_text_props(color=TEXT_MAIN, fontweight="bold")
        cell.set_edgecolor(TEXT_MUTED)

    # Style rows — highlight KMV row
    for i, model in enumerate(models):
        for j in range(len(metrics_display)):
            cell = table[i + 1, j]
            if "KMV" in model:
                cell.set_facecolor('#450a0a') # Dark red background for benchmark
                cell.set_text_props(color=TEXT_MAIN)
            elif i % 2 == 0:
                cell.set_facecolor(BG_SURFACE)
                cell.set_text_props(color=TEXT_MAIN)
            else:
                cell.set_facecolor('none')
                cell.set_text_props(color=TEXT_MAIN)
            cell.set_edgecolor(TEXT_MUTED)

        # Row label
        row_cell = table[i + 1, -1]
        if "KMV" in model:
            row_cell.set_facecolor(ACCENT_RED)
            row_cell.set_text_props(color=TEXT_MAIN, fontweight="bold")
        else:
            row_cell.set_facecolor(BG_SURFACE)
            row_cell.set_text_props(color=TEXT_MAIN)
        row_cell.set_edgecolor(TEXT_MUTED)

    # Highlight best AUC
    auc_values = [all_metrics[m].get("auc", 0) for m in models]
    best_auc_idx = np.argmax(auc_values)
    table[best_auc_idx + 1, 0].set_text_props(fontweight="bold", color=ACCENT_GREEN)

    ax.set_title("Comprehensive Model Comparison: ML vs KMV (Merton)\nBIST Financial Distress Prediction (2018-2024)",
                 fontsize=15, fontweight="bold", pad=20, color=TEXT_MAIN)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Summary table -> {filename}")


def main():
    print("=" * 65)
    print("  KMV (Merton) vs ML Models -- Comparison Pipeline")
    print("=" * 65)

    # ── Load dataset ──────────────────────────────────────────────────────
    if not DATASET_FILENAME.exists():
        print("Error: Dataset not found. Run build_dataset.py first.")
        return

    df = pd.read_csv(DATASET_FILENAME)
    print(f"  Dataset loaded: {len(df)} records")

    # ── Run KMV Model ─────────────────────────────────────────────────────
    print("\n[1] Running KMV (Merton) model...")
    kmv_result = run_kmv_model(df, target_col=TARGET)

    if not kmv_result["metrics"]:
        print("  KMV model failed. Skipping comparison.")
        return

    # ── Load ML CV Results ────────────────────────────────────────────────
    print("\n[2] Loading ML model results...")
    cv_df = load_cv_results()
    if cv_df is None:
        return

    # Build ML metrics dict (No SMOTE versions)
    ml_metrics = {}
    for _, row in cv_df[cv_df["SMOTE"] == "No SMOTE"].iterrows():
        model_name = f"{row['Model']} (No SMOTE)"
        ml_metrics[model_name] = {
            "accuracy": row["accuracy_mean"],
            "precision": row["precision_mean"],
            "recall": row["recall_mean"],
            "f1": row["f1_mean"],
            "auc": row["auc_mean"],
            "mcc": row["mcc_mean"],
        }

    print(f"  ML models loaded: {list(ml_metrics.keys())}")

    # ── Generate ROC data for ML models ───────────────────────────────────
    print("\n[3] Computing ROC curves for all models...")

    # For ROC overlay, we need to retrain and get probabilities
    from sklearn.model_selection import cross_val_predict
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import roc_curve as sk_roc_curve, roc_auc_score
    from src.ml_models import build_model_configs
    from src.feature_selection import select_features_freg

    available_features = [f for f in CANDIDATE_FEATURES if f in df.columns]
    X = df[available_features].copy()
    y = df[TARGET].copy()

    # Feature selection
    sel_features, _, _ = select_features_freg(X, y, k=FEATURE_SELECTION_K)
    X_sel = X[sel_features]

    # Scale
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_sel),
        columns=sel_features, index=X_sel.index,
    )

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    model_configs = build_model_configs(scale_pos_weight=n_neg / n_pos)
    ml_roc_data = {}

    # Get best params from CV results
    for _, row in cv_df[cv_df["SMOTE"] == "No SMOTE"].iterrows():
        model_name = row["Model"]
        if model_name not in model_configs:
            continue

        base_model, _ = model_configs[model_name]
        try:
            best_params = json.loads(row["best_params"])
        except (json.JSONDecodeError, TypeError):
            best_params = {}

        from sklearn.base import clone
        model = clone(base_model)
        model.set_params(**best_params)

        # Cross-val predict for ROC
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

        y_proba = np.zeros(len(y))
        for train_idx, test_idx in skf.split(X_scaled, y):
            X_train = X_scaled.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X_scaled.iloc[test_idx]

            m = clone(base_model)
            m.set_params(**best_params)
            m.fit(X_train, y_train)
            y_proba[test_idx] = m.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = sk_roc_curve(y, y_proba)
        auc_val = roc_auc_score(y, y_proba)
        ml_roc_data[model_name] = {"fpr": fpr, "tpr": tpr, "auc": auc_val}
        print(f"    {model_name}: AUC={auc_val:.4f}")

    # KMV ROC data
    kmv_roc = None
    if kmv_result["roc_curve"]:
        kmv_roc = {
            "fpr": kmv_result["roc_curve"]["fpr"],
            "tpr": kmv_result["roc_curve"]["tpr"],
            "auc": kmv_result["metrics"]["auc"],
        }

    # ── Generate Comparison Charts ────────────────────────────────────────
    print("\n[4] Generating comparison visualizations...")

    # All metrics for comparison (No SMOTE ML + KMV)
    all_metrics = dict(ml_metrics)
    all_metrics["KMV (Merton)"] = kmv_result["metrics"]

    # ML AUC dict for bar chart
    ml_auc = {m: ml_metrics[m]["auc"] for m in ml_metrics}

    plot_auc_comparison(ml_auc, kmv_result["metrics"])
    plot_roc_overlay(ml_roc_data, kmv_roc)
    plot_radar_comparison(all_metrics)
    plot_summary_table(all_metrics)

    # ── Save comparison results ───────────────────────────────────────────
    print("\n[5] Saving comparison results...")
    comparison_data = {
        "kmv_metrics": kmv_result["metrics"],
        "ml_metrics": ml_metrics,
        "improvement_over_kmv": {},
    }

    kmv_auc = kmv_result["metrics"]["auc"]
    for model_name, metrics in ml_metrics.items():
        ml_auc_val = metrics["auc"]
        improvement = ml_auc_val - kmv_auc
        comparison_data["improvement_over_kmv"][model_name] = {
            "auc_difference": round(improvement, 4),
            "auc_improvement_pct": round((improvement / kmv_auc) * 100, 2),
        }

    comp_path = OUTPUTS_DIR / "kmv_vs_ml_comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2, default=str)
    print(f"  Comparison saved to {comp_path}")

    # ── Print Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  COMPARISON SUMMARY")
    print("=" * 65)
    print(f"  {'Model':<35s} {'AUC':>8s} {'F1':>8s} {'MCC':>8s}")
    print(f"  {'-'*35} {'-'*8} {'-'*8} {'-'*8}")

    for model_name, metrics in sorted(all_metrics.items(),
                                       key=lambda x: x[1].get("auc", 0),
                                       reverse=True):
        marker = " <-- BEST" if metrics["auc"] == max(m["auc"] for m in all_metrics.values()) else ""
        if "KMV" in model_name:
            marker = " <-- BENCHMARK"
        print(f"  {model_name:<35s} {metrics['auc']:>8.4f} "
              f"{metrics['f1']:>8.4f} {metrics['mcc']:>8.4f}{marker}")

    print(f"\n  KMV AUC:  {kmv_auc:.4f}")
    best_ml = max(ml_metrics.items(), key=lambda x: x[1]["auc"])
    improvement = best_ml[1]["auc"] - kmv_auc
    print(f"  Best ML:  {best_ml[0]} (AUC={best_ml[1]['auc']:.4f})")
    print(f"  Improvement over KMV: +{improvement:.4f} "
          f"(+{(improvement/kmv_auc)*100:.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    main()
