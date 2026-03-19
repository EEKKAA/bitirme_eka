"""
Visualization module to generate thesis-ready plots.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
from config import PLOTS_DIR

def plot_roc_curve(model_dict, X_test, y_test, filename="roc_curve.png"):
    """Plots ROC curves for all models in one figure."""
    plt.figure(figsize=(8, 6))
    
    for name, model in model_dict.items():
        if model is None: 
            continue
            
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
        
        try:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.2f})")
        except ValueError:
            # Handles edge cases where y_test might have only 1 class
            plt.plot([0], [0], lw=2, label=f"{name} (AUC error)")
            
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300)
    plt.close()

def plot_confusion_matrix(cm, model_name, filename):
    """Plots the confusion matrix using seaborn heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=['Non-Bankrupt', 'Bankrupt'],
                yticklabels=['Non-Bankrupt', 'Bankrupt'])
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300)
    plt.close()

def plot_feature_importance(model, features, model_name, filename):
    """Plots the feature importances for tree-based models."""
    if not hasattr(model, 'feature_importances_'):
        return
        
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Take care of potential length mismatches (e.g. if categorical encodings were used, though we are restricting to continuous ratios here)
    sorted_features = [features[i] for i in indices[:len(features)]]
    sorted_importances = importances[indices][:len(features)]
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Feature Importances: {model_name}")
    sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis")
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300)
    plt.close()

def plot_ratio_distributions(df, features, target_col, filename="distributions.png"):
    """Plots boxplots to compare ratio distributions broken down by target label."""
    # Select only available features
    available_features = [f for f in features if f in df.columns]
    num_features = len(available_features)
    
    if num_features == 0:
        return
        
    cols_plot = 2
    rows_plot = int(np.ceil(num_features / cols_plot))
    
    fig, axes = plt.subplots(rows_plot, cols_plot, figsize=(14, 4 * rows_plot))
    # Standardize axes to a list even if it's 1-D
    axes_flat = axes.flatten() if num_features > 1 else [axes]
    
    for i, col in enumerate(available_features):
        sns.boxplot(x=target_col, y=col, data=df, ax=axes_flat[i], palette="Set2")
        axes_flat[i].set_title(f'Distribution of {col}')
        # Log scale might be useful for highly skewed financial ratios, 
        # but since we winsorized the data, linear scale is usually acceptable.
        
    # Hide any unused subplots
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
        
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300)
    plt.close()

def plot_correlation_heatmap(df, features, filename="correlation_heatmap.png"):
    """Generates a correlation heatmap between the specified features."""
    cols = [f for f in features if f in df.columns]
    if not cols:
        return
        
    plt.figure(figsize=(10, 8))
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm',
                vmax=1, vmin=-1, center=0, square=True, linewidths=.5,
                cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / filename, dpi=300)
    plt.close()
