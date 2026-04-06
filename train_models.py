"""
Main pipeline — Financial Distress Prediction for BIST Companies.
Buyukarikan & Buyukarikan (2025) methodology.

Steps:
  1. Load dataset
  2. Preprocess (imputation)
  3. 10-fold stratified group CV (company-level splits)
     with MinMax, feature selection, threshold optimization inside folds
  4. Hyperparameter tuning (RandomizedSearchCV inside each fold)
  5. Model comparison
  6. Best model selection (primary: AUC, secondary: F1)
  7. Retrain best model on full data
  8. SHAP explainability analysis
  9. Save outputs (model .pkl, selected_ratios.json, plots, metrics)
"""
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

sys.stdout.reconfigure(line_buffering=True)

from config import (
    DATASET_FILENAME, CV_RESULTS_FILENAME, BEST_MODEL_FILENAME,
    SELECTED_RATIOS_FILENAME, PLOTS_DIR,
    CANDIDATE_FEATURES, TARGET, RANDOM_STATE,
    PRIMARY_METRIC, SECONDARY_METRIC, FEATURE_SELECTION_K,
)
from src.preprocessing import preprocess_data
from src.ml_models import build_model_configs
from src.evaluation import cross_validate_model
from src.feature_selection import select_features_freg
from src.shap_analysis import run_shap_analysis
from src.visualization import (
    plot_cv_comparison, plot_roc_curve, plot_confusion_matrix,
    plot_feature_importance, plot_performance_table,
    plot_oof_threshold_analysis,
)

# Redirect output to log file as well
import io

LOG_FILE = DATASET_FILENAME.parent / "train_log.txt"


def main():
    # Tee output to both console and log file
    log_f = open(LOG_FILE, "w", encoding="utf-8")

    def tee_print(*args, **kwargs):
        import builtins
        builtins.print(*args, **kwargs)
        kwargs.pop("file", None)
        print_str = " ".join(str(a) for a in args)
        log_f.write(print_str + "\n")
        log_f.flush()

    print = tee_print

    print("=" * 65)
    print("  Financial Distress Prediction — BIST Companies")
    print("  Buyukarikan & Buyukarikan (2025) Methodology")
    print("=" * 65)

    # ── STEP 1: Load dataset ──────────────────────────────────────────────
    if not DATASET_FILENAME.exists():
        print(f"\nError: Dataset '{DATASET_FILENAME}' not found.")
        print("Run build_labeling_dataset.py first.")
        log_f.close()
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
        log_f.close()
        return

    if TARGET not in df.columns:
        print(f"Error: Target column '{TARGET}' not found.")
        log_f.close()
        return

    X = df[available_features].copy()
    y = df[TARGET].copy()
    groups = df["company"].copy() if "company" in df.columns else None

    distress_count = int(y.sum())
    healthy_count = len(y) - distress_count
    print(f"  Features: {len(available_features)}")
    print(f"  Distressed: {distress_count} ({distress_count/len(y):.1%})")
    print(f"  Healthy: {healthy_count} ({healthy_count/len(y):.1%})")
    if groups is not None:
        print(f"  Companies: {groups.nunique()} (StratifiedGroupKFold)")

    # ── STEP 3-5: Cross-validated model training ──────────────────────────
    model_configs = build_model_configs()
    cv_results = {}

    for model_name, (model, param_grid) in model_configs.items():
        print(f"\n  --- {model_name} ---")
        result = cross_validate_model(
            model, param_grid, X, y,
            groups=groups,
            model_name=model_name,
        )
        cv_results[model_name] = result

        mean = result["mean_metrics"]
        std = result["std_metrics"]
        mean_opt = result["mean_metrics_opt"]
        print(f"\n    Mean Metrics (t=0.5):")
        for metric in ["accuracy", "precision", "recall", "f1", "auc", "mcc"]:
            print(f"      {metric:>10s}: {mean[metric]:.4f} +/- {std[metric]:.4f}")
        print(f"    Optimized Threshold: {result['optimal_threshold']:.2f}")
        print(f"    Mean Metrics (opt_t):")
        for metric in ["accuracy", "precision", "recall", "f1", "auc", "mcc"]:
            print(f"      {metric:>10s}: {mean_opt[metric]:.4f}")
        print(f"    Best Params: {result['best_params']}")

    # ── Comparison Table ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  MODEL COMPARISON TABLE")
    print("=" * 80)
    print(f"  {'Model':<25s} {'AUC':>8s} {'F1':>8s} {'MCC':>8s} "
          f"{'OOF_F1':>8s} {'OOF_MCC':>8s} {'OOF_t':>6s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for model_name, result in cv_results.items():
        m = result["mean_metrics"]
        oof = result["oof_metrics"]
        oof_t = result["oof_threshold"]
        print(f"  {model_name:<25s} "
              f"{m['auc']:>8.4f} {m['f1']:>8.4f} {m['mcc']:>8.4f} "
              f"{oof['f1']:>8.4f} {oof['mcc']:>8.4f} {oof_t:>6.2f}")
    print("=" * 80)

    # Select best model (primary: AUC, secondary: F1)
    best_name = max(cv_results, key=lambda k: (
        cv_results[k]["mean_metrics"]["auc"],
        cv_results[k]["mean_metrics"]["f1"]
    ))
    best_result = cv_results[best_name]
    print(f"\n  Best: {best_name}")
    print(f"    AUC: {best_result['mean_metrics']['auc']:.4f}")
    print(f"    F1:  {best_result['mean_metrics']['f1']:.4f}")
    print(f"    OOF Threshold: {best_result['oof_threshold']:.2f}")
    print(f"    OOF F1: {best_result['oof_metrics']['f1']:.4f}")
    print(f"    OOF MCC: {best_result['oof_metrics']['mcc']:.4f}")

    # ── STEP 6: Save CV results ───────────────────────────────────────────
    print("\n[Step 6] Saving cross-validation results...")
    cv_rows = []
    for model_name, result in cv_results.items():
        row = {"Model": model_name}
        for m in ["accuracy", "precision", "recall", "f1", "auc", "mcc"]:
            row[f"{m}_mean"] = result["mean_metrics"][m]
            row[f"{m}_std"] = result["std_metrics"][m]
            row[f"{m}_opt_mean"] = result["mean_metrics_opt"][m]
        row["optimal_threshold"] = result["optimal_threshold"]
        row["oof_threshold"] = result["oof_threshold"]
        for m in ["accuracy", "precision", "recall", "f1", "auc", "mcc"]:
            row[f"{m}_oof"] = result["oof_metrics"][m]
        row["best_params"] = json.dumps(result["best_params"])
        cv_rows.append(row)

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(CV_RESULTS_FILENAME, index=False)
    print(f"  Saved to {CV_RESULTS_FILENAME}")

    # ── STEP 7: Retrain best model on full data ───────────────────────────
    print(f"\n[Step 7] Retraining {best_name} on full dataset...")

    # Impute any remaining NaN with global median (full data — no CV here)
    from src.preprocessing import winsorize
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

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

    # Retrain with best hyperparameters
    from sklearn.base import clone
    model_configs_all = build_model_configs()
    base_model = model_configs_all[best_name][0]
    best_model = clone(base_model)
    best_params = best_result["best_params"]
    best_model.set_params(**best_params)
    best_model.fit(X_scaled, y)

    # Save model
    joblib.dump(best_model, BEST_MODEL_FILENAME)
    print(f"  Model saved to {BEST_MODEL_FILENAME}")

    # ── STEP 8: Visualizations ────────────────────────────────────────────
    print("\n[Step 8] Generating visualizations...")

    plot_cv_comparison(cv_results, filename="cv_comparison.png")
    print("  [OK] CV comparison chart")

    plot_performance_table(cv_results, filename="performance_table.png")
    print("  [OK] Performance table")

    plot_feature_importance(
        best_model, sel_features, best_name,
        f"fi_{best_name.replace(' ', '_').lower()}.png"
    )
    print("  [OK] Feature importance")

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

    # OOF threshold analysis plot
    oof_preds = best_result["oof_predictions"]
    plot_oof_threshold_analysis(
        oof_preds["y_true"], oof_preds["y_prob"],
        best_result["oof_threshold"], best_name,
        f"oof_threshold_{best_name.replace(' ', '_').lower()}.png"
    )
    print("  [OK] OOF threshold analysis")

    # OOF confusion matrix (at optimal threshold)
    from sklearn.metrics import confusion_matrix as cm_func
    oof_cm = cm_func(
        oof_preds["y_true"],
        (oof_preds["y_prob"] >= best_result["oof_threshold"]).astype(int)
    ).tolist()
    plot_confusion_matrix(
        oof_cm, f"{best_name} (OOF t={best_result['oof_threshold']:.2f})",
        f"cm_oof_{best_name.replace(' ', '_').lower()}.png"
    )
    print("  [OK] OOF confusion matrix")

    # ── STEP 9: SHAP Analysis ────────────────────────────────────────────
    print("\n[Step 9] SHAP explainability analysis...")
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
    print(f"  AUC: {best_result['mean_metrics']['auc']:.4f} +/- "
          f"{best_result['std_metrics']['auc']:.4f}")
    print(f"  F1:  {best_result['mean_metrics']['f1']:.4f} +/- "
          f"{best_result['std_metrics']['f1']:.4f}")
    print(f"  MCC: {best_result['mean_metrics']['mcc']:.4f} +/- "
          f"{best_result['std_metrics']['mcc']:.4f}")
    print(f"  OOF Threshold: {best_result['oof_threshold']:.2f} "
          f"(F1={best_result['oof_metrics']['f1']:.4f}, "
          f"MCC={best_result['oof_metrics']['mcc']:.4f})")
    print(f"\n  Outputs saved to: {CV_RESULTS_FILENAME.parent}")
    print("=" * 65)

    log_f.close()


if __name__ == "__main__":
    main()
