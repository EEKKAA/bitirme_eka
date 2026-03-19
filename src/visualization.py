"""
Visualization module for the financial distress prediction pipeline.
Generates thesis-ready plots: ROC curves, confusion matrices,
CV comparison, and feature importance.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from config import PLOTS_DIR

# --- GLOBAL STYLING TO MATCH HTML PRESENTATION ---
BG_DARK = '#0b1120'
BG_SURFACE = '#1e293b'
TEXT_MAIN = '#f8fafc'
TEXT_MUTED = '#cbd5e1'
ACCENT_BLUE = '#3b82f6'
ACCENT_GREEN = '#10b981'
ACCENT_RED = '#ef4444'

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': 'none', # transparent to blend with html
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


def plot_cv_comparison(cv_results: dict, filename: str = "cv_comparison.png"):
    """
    Bar chart comparing model performance across CV folds (mean ± std).

    Args:
        cv_results: Dict of {model_name: {'mean_metrics': {...}, 'std_metrics': {...}}}
    """
    metrics_to_plot = ["auc", "f1", "accuracy", "mcc"]
    model_names = list(cv_results.keys())
    n_metrics = len(metrics_to_plot)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    colors = sns.color_palette("viridis", len(model_names))

    for ax, metric in zip(axes, metrics_to_plot):
        means = [cv_results[m]["mean_metrics"].get(metric, 0) for m in model_names]
        stds = [cv_results[m]["std_metrics"].get(metric, 0) for m in model_names]

        bars = ax.bar(model_names, means, yerr=stds, capsize=5,
                      color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(metric.upper(), fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")

        # Add value labels
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.02,
                    f"{mean:.3f}", ha="center", va="bottom", fontsize=9)

        ax.tick_params(axis="x", rotation=15)

    plt.suptitle("Model Comparison — Stratified 10-Fold CV", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curve(models_data: dict, filename: str = "roc_curve.png"):
    """
    Plots ROC curves for all models.

    Args:
        models_data: Dict of {model_name: (model, X_test, y_test)}
    """
    plt.figure(figsize=(8, 6))

    for name, (model, X_test, y_test) in models_data.items():
        if model is None:
            continue
        y_prob = (model.predict_proba(X_test)[:, 1]
                  if hasattr(model, "predict_proba")
                  else model.predict(X_test))
        try:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")
        except ValueError:
            plt.plot([0], [0], lw=2, label=f"{name} (AUC error)")

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Financial Distress Prediction")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300)
    plt.close()


def plot_confusion_matrix(cm, model_name: str, filename: str):
    """Plots the confusion matrix as a heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Healthy", "Distress"],
                yticklabels=["Healthy", "Distress"])
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300)
    plt.close()


def plot_feature_importance(model, features: list, model_name: str,
                            filename: str):
    """Plots feature importance for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    sorted_features = [features[i] for i in indices[:len(features)]]
    sorted_importances = importances[indices][:len(features)]

    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances: {model_name}")
    sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis")
    plt.xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300)
    plt.close()


def plot_performance_table(cv_results: dict, filename: str = "performance_table.png"):
    """
    Renders a performance summary table as an image.

    Args:
        cv_results: Dict of {model_name: {'mean_metrics': {...}, 'std_metrics': {...}}}
    """
    metrics = ["accuracy", "precision", "recall", "f1", "auc", "mcc"]
    rows = []
    for model_name, result in cv_results.items():
        row = {"Model": model_name}
        for m in metrics:
            mean_val = result["mean_metrics"].get(m, 0)
            std_val = result["std_metrics"].get(m, 0)
            row[m.upper()] = f"{mean_val:.4f} ± {std_val:.4f}"
        rows.append(row)

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, 2 + len(rows) * 0.5))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Style header
    for key, cell in table.get_celld().items():
        if key[0] == 0:
            cell.set_facecolor(BG_SURFACE)
            cell.set_text_props(color=TEXT_MAIN, fontweight="bold")
        else:
            cell.set_facecolor(BG_SURFACE if key[0] % 2 == 0 else 'none')
            cell.set_text_props(color=TEXT_MAIN)
        cell.set_edgecolor(TEXT_MUTED)

    plt.title("Model Performance — Stratified 10-Fold CV", fontweight="bold",
              pad=20, color=TEXT_MAIN)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300, bbox_inches="tight", facecolor='none')
    plt.close()
