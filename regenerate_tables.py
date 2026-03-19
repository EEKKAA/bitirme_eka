import pandas as pd
from config import CV_RESULTS_FILENAME
from src.visualization import plot_performance_table, plot_cv_comparison
import json

def regenerate():
    print("Loading", CV_RESULTS_FILENAME)
    df = pd.read_csv(CV_RESULTS_FILENAME)
    all_cv_results = {}

    for mode in df['SMOTE'].unique():
        subset = df[df['SMOTE'] == mode]
        cv_res = {}
        for _, row in subset.iterrows():
            model_name = row['Model']
            mean_metrics = {m: row[f"{m}_mean"] for m in ["accuracy", "precision", "recall", "f1", "auc", "mcc"]}
            std_metrics = {m: row[f"{m}_std"] for m in ["accuracy", "precision", "recall", "f1", "auc", "mcc"]}
            cv_res[model_name] = {"mean_metrics": mean_metrics, "std_metrics": std_metrics}
        all_cv_results[mode] = cv_res

    print("Generating cv_comparison plots...")
    for mode_label, cv_res in all_cv_results.items():
        suffix = mode_label.replace(" ", "_").lower()
        plot_cv_comparison(cv_res, filename=f"cv_comparison_{suffix}.png")
        if mode_label == "No SMOTE":
            plot_cv_comparison(cv_res, filename="cv_comparison.png") # Main one used in presentation

    print("Generating performance_table.png...")
    combined_results = {}
    for mode_label, cv_res in all_cv_results.items():
        for model_name, result in cv_res.items():
            combined_results[f"{model_name} ({mode_label})"] = result
    plot_performance_table(combined_results, filename="performance_table.png")
    
    print("Optimization Complete.")

if __name__ == "__main__":
    regenerate()
