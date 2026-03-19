"""
Entry point wrapper script to train machine learning models, visualize and evaluate them.
"""
import pandas as pd
import json
from sklearn.model_selection import train_test_split

from config import (
    DATASET_FILENAME, FEATURES, TARGET, RANDOM_STATE, TEST_SIZE, 
    METRICS_FILENAME, MODEL_COMP_FILENAME
)
from src.preprocessing import preprocess_data
from src.ml_models import train_logistic, train_random_forest, train_xgboost
from src.evaluation import evaluate_model
from src.visualization import (
    plot_roc_curve, plot_confusion_matrix, plot_feature_importance,
    plot_ratio_distributions, plot_correlation_heatmap
)

def main():
    print("Starting ML pipeline...")
    
    if not DATASET_FILENAME.exists():
        print(f"Error: Dataset '{DATASET_FILENAME}' not found. Run build_dataset.py first.")
        return
        
    print(f"Loading dataset from {DATASET_FILENAME}...")
    df = pd.read_csv(DATASET_FILENAME)
    
    print("Generating exploratory plots...")
    plot_ratio_distributions(df, FEATURES, TARGET)
    plot_correlation_heatmap(df, FEATURES)
    
    print("Preprocessing data...")
    df_processed = preprocess_data(df)
    
    available_features = [f for f in FEATURES if f in df_processed.columns]
    if not available_features:
        print("Error: No features available for training.")
        return
        
    if TARGET not in df_processed.columns:
        print(f"Error: Target column '{TARGET}' not found.")
        return
        
    X = df_processed[available_features]
    y = df_processed[TARGET]
    
    print("Splitting dataset into train/test sets...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
    except ValueError:
        print("Warning: Insufficient class distribution for stratified split. Using standard split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    
    models = {}
    print("Training Logistic Regression...")
    models['Logistic Regression'] = train_logistic(X_train, y_train)
    
    print("Training Random Forest...")
    models['Random Forest'] = train_random_forest(X_train, y_train)
    
    print("Training XGBoost...")
    models['XGBoost'] = train_xgboost(X_train, y_train)
    
    metrics = {}
    print("Evaluating models...")
    for name, model in models.items():
        if model is None: 
            continue
            
        print(f" => {name}")
        results = evaluate_model(model, X_test, y_test)
        metrics[name] = results
        
        if 'confusion_matrix' in results:
            clean_name = name.replace(' ', '_').lower()
            plot_confusion_matrix(results['confusion_matrix'], name, f"cm_{clean_name}.png")
            
        # Plot feature importance
        plot_feature_importance(model, available_features, name, f"fi_{clean_name}.png")
        
    print("Generating ROC Curve...")
    plot_roc_curve(models, X_test, y_test)
    
    print("Saving metrics...")
    metrics_df = pd.DataFrame(metrics).T
    
    # Drop complex object (list) before saving to csv
    if 'confusion_matrix' in metrics_df.columns:
        metrics_df_out = metrics_df.drop(columns=['confusion_matrix'])
    else:
        metrics_df_out = metrics_df
        
    metrics_df_out.to_csv(MODEL_COMP_FILENAME)
    
    with open(METRICS_FILENAME, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"\nML Pipeline complete!\nModels comparison saved to: {MODEL_COMP_FILENAME}\nFull metrics saved to: {METRICS_FILENAME}")

if __name__ == "__main__":
    main()
