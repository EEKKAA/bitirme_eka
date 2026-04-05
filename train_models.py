"""
Main pipeline — Financial Distress Prediction for BIST Companies.
Büyükarıkan & Büyükarıkan (2025) methodology.

Steps:
  1. Load dataset
  2. Preprocess (imputation + winsorization)
  3. 10-fold stratified CV (with MinMax, SMOTE, feature selection inside folds)
  4. Hyperparameter tuning (GridSearchCV inside each fold)
  5. Model comparison
  6. Best model selection (primary: AUC, secondary: F1)
  7. Retrain best model on full data
  8. SHAP explainability analysis
  9. Save outputs (model .pkl, selected_ratios.json, plots, metrics)
"""
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from config import (
    DATASET_FILENAME, CV_RESULTS_FILENAME, BEST_MODEL_FILENAME,
    SELECTED_RATIOS_FILENAME, PLOTS_DIR,
    CANDIDATE_FEATURES, TARGET, RANDOM_STATE,
    PRIMARY_METRIC, SECONDARY_METRIC, FEATURE_SELECTION_K,
    SMOTE_THRESHOLD,
)
from src.preprocessing import preprocess_data
from src.ml_models import build_model_configs
from src.evaluation import cross_validate_model
from src.feature_selection import select_features_freg
from src.shap_analysis import run_shap_analysis
from src.visualization import (
    plot_cv_comparison, plot_roc_curve, plot_confusion_matrix,
    plot_feature_importance, plot_performance_table,
)

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


def main():
    print("=" * 65)
    print("  Financial Distress Prediction — BIST Companies")
    print("  Büyükarıkan & Büyükarıkan (2025) Methodology")
    print("=" * 65)

    # ── STEP 1: Load dataset ──────────────────────────────────────────────
    if not DATASET_FILENAME.exists():
        print(f"\nError: Dataset '{DATASET_FILENAME}' not found.")
        print("Run build_dataset.py first.")
        return

    print("\n[Step 1] Loading dataset...")
    df = pd.read_csv(DATASET_FILENAME)
    print(f"  Records: {len(df)}")

    # ── STEP 2: Preprocess ────────────────────────────────────────────────
    print("\n[Step 2] Preprocessing...")
    df = preprocess_data(df)

    # Extract features and target
    available_features = [r for r in CANDIDATE_FEATURES if r in df.columns]
    if not available_features:
        print("Error: No feature columns found in dataset.")
        return

    if TARGET not in df.columns:
        print(f"Error: Target column '{TARGET}' not found.")
        return

    X = df[available_features].copy()
    y = df[TARGET].copy()

    distress_count = int(y.sum())
    healthy_count = len(y) - distress_count
    print(f"  Features: {len(available_features)}")
    print(f"  Distressed: {distress_count} ({distress_count/len(y):.1%})")
    print(f"  Healthy: {healthy_count} ({healthy_count/len(y):.1%})")

    # ── STEP 3-5: Cross-validated model training ──────────────────────────
    model_configs = build_model_configs()

    # Run both modes: without SMOTE and with SMOTE
    all_cv_results = {}
    for smote_label, smote_flag in [("No SMOTE", False), ("SMOTE", True)]:
        print(f"\n{'='*50}")
        print(f"  MODE: {smote_label}")
        print(f"{'='*50}")
        cv_results = {}
        for model_name, (model, param_grid) in model_configs.items():
            display_name = f"{model_name}"
            print(f"\n  --- {display_name} ---")
            result = cross_validate_model(
                model, param_grid, X, y,
                model_name=display_name,
                force_smote=smote_flag,
            )
            cv_results[model_name] = result

            mean = result["mean_metrics"]
            std = result["std_metrics"]
            print(f"\n    Mean Metrics:")
            for metric in ["accuracy", "precision", "recall", "f1", "auc", "mcc"]:
                print(f"      {metric:>10s}: {mean[metric]:.4f} +/- {std[metric]:.4f}")
            print(f"    Best Params: {result['best_params']}")

        all_cv_results[smote_label] = cv_results

    # ── Comparison Table ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SMOTE COMPARISON TABLE")
    print("=" * 70)
    print(f"  {'Model':<25s} {'Mode':<12s} {'AUC':>8s} {'F1':>8s} {'MCC':>8s}")
    print(f"  {'-'*25} {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
    for mode_label, cv_res in all_cv_results.items():
        for model_name, result in cv_res.items():
            m = result["mean_metrics"]
            print(f"  {model_name:<25s} {mode_label:<12s} "
                  f"{m['auc']:>8.4f} {m['f1']:>8.4f} {m['mcc']:>8.4f}")
    print("=" * 70)

    # Select best overall (from both modes)
    best_name = None
    best_mode = None
    best_auc = -1
    best_f1 = -1
    for mode_label, cv_res in all_cv_results.items():
        for model_name, result in cv_res.items():
            auc_val = result["mean_metrics"]["auc"]
            f1_val = result["mean_metrics"]["f1"]
            if (auc_val, f1_val) > (best_auc, best_f1):
                best_auc = auc_val
                best_f1 = f1_val
                best_name = model_name
                best_mode = mode_label

    cv_results = all_cv_results[best_mode]
    best_result = cv_results[best_name]
    print(f"\n  Best: {best_name} ({best_mode})")
    print(f"    AUC: {best_result['mean_metrics']['auc']:.4f}")
    print(f"    F1:  {best_result['mean_metrics']['f1']:.4f}")

    # ── STEP 6: Save CV results ───────────────────────────────────────────
    print("\n[Step 6] Saving cross-validation results...")
    cv_rows = []
    for mode_label, cv_res in all_cv_results.items():
        for model_name, result in cv_res.items():
            row = {"Model": model_name, "SMOTE": mode_label}
            for m in ["accuracy", "precision", "recall", "f1", "auc", "mcc"]:
                row[f"{m}_mean"] = result["mean_metrics"][m]
                row[f"{m}_std"] = result["std_metrics"][m]
            row["best_params"] = json.dumps(result["best_params"])
            cv_rows.append(row)


    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(CV_RESULTS_FILENAME, index=False)
    print(f"  Saved to {CV_RESULTS_FILENAME}")

    # ── STEP 7: Best model already selected above ───────────────────────────
    print(f"\n[Step 7] Best model: {best_name} ({best_mode})")
    print(f"    {PRIMARY_METRIC.upper()}: "
          f"{best_result['mean_metrics'][PRIMARY_METRIC]:.4f}")
    print(f"    {SECONDARY_METRIC.upper()}: "
          f"{best_result['mean_metrics'][SECONDARY_METRIC]:.4f}")

    # ── STEP 8: Retrain best model on full data ───────────────────────────
    print("\n[Step 8] Retraining best model on full dataset...")

    # Winsorize full data (no leakage concern — retraining on everything)
    from src.preprocessing import winsorize
    X_w = winsorize(X)

    # Feature selection on full data
    sel_features, f_scores, _ = select_features_freg(X_w, y, k=FEATURE_SELECTION_K)
    X_selected = X_w[sel_features]

    # MinMax scaling
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_selected),
        columns=sel_features, index=X_selected.index,
    )

    # SMOTE if best mode was SMOTE
    if best_mode == "SMOTE" and SMOTE_AVAILABLE:
        smote = SMOTE(random_state=RANDOM_STATE)
        X_final, y_final = smote.fit_resample(X_scaled, y)
        print(f"  SMOTE applied (best mode: {best_mode})")
    else:
        X_final, y_final = X_scaled, y
        print(f"  No SMOTE (best mode: {best_mode})")

    # Retrain with best hyperparameters
    from sklearn.base import clone
    model_configs_all = build_model_configs()
    base_model = model_configs_all[best_name][0]
    best_model = clone(base_model)
    best_params = best_result["best_params"]
    best_model.set_params(**best_params)
    best_model.fit(X_final, y_final)

    # Save model
    joblib.dump(best_model, BEST_MODEL_FILENAME)
    print(f"  Model saved to {BEST_MODEL_FILENAME}")

    # ── STEP 9: Visualizations ────────────────────────────────────────────
    print("\n[Step 9] Generating visualizations...")

    # CV comparison charts for each mode
    for mode_label, cv_res in all_cv_results.items():
        suffix = mode_label.replace(" ", "_").lower()
        plot_cv_comparison(cv_res, filename=f"cv_comparison_{suffix}.png")
    print("  [OK] CV comparison charts (No SMOTE + SMOTE)")

    # Performance table for all results combined
    combined_results = {}
    for mode_label, cv_res in all_cv_results.items():
        for model_name, result in cv_res.items():
            combined_results[f"{model_name} ({mode_label})"] = result
    plot_performance_table(combined_results, filename="performance_table.png")
    print("  [OK] Performance table")

    # Feature importance (for tree-based best model)
    plot_feature_importance(
        best_model, sel_features, best_name,
        f"fi_{best_name.replace(' ', '_').lower()}.png"
    )
    print("  [OK] Feature importance")

    # Confusion matrix for the best fold
    best_fold = max(
        best_result["fold_metrics"],
        key=lambda f: f.get("auc", 0)
    )
    if "confusion_matrix" in best_fold:
        plot_confusion_matrix(
            best_fold["confusion_matrix"], best_name,
            f"cm_{best_name.replace(' ', '_').lower()}.png"
        )
        print("  [OK] Confusion matrix")

    # ── STEP 10: SHAP Analysis ────────────────────────────────────────────
    print("\n[Step 10] SHAP explainability analysis...")
    shap_result = run_shap_analysis(
        best_model, X_scaled, sel_features, model_name=best_name
    )

    if shap_result["feature_ranking"]:
        print("\n  Top Feature Ranking (by SHAP):")
        for item in shap_result["feature_ranking"][:10]:
            print(f"    {item['rank']:2d}. {item['feature']:40s} "
                  f"(|SHAP| = {item['mean_abs_shap']:.4f})")

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  Pipeline Complete!")
    print(f"  Best Model: {best_name}")
    print(f"  AUC: {best_result['mean_metrics']['auc']:.4f} ± "
          f"{best_result['std_metrics']['auc']:.4f}")
    print(f"  F1:  {best_result['mean_metrics']['f1']:.4f} ± "
          f"{best_result['std_metrics']['f1']:.4f}")
    print(f"  MCC: {best_result['mean_metrics']['mcc']:.4f} ± "
          f"{best_result['std_metrics']['mcc']:.4f}")
    print(f"\n  Outputs saved to: {CV_RESULTS_FILENAME.parent}")
    print("=" * 65)


if __name__ == "__main__":
    main()
