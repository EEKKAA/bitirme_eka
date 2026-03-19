"""
Evaluation module to assess ML model performance.
"""
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix
)
import numpy as np

def evaluate_model(model, X_test, y_test) -> dict:
    """
    Evaluates a trained model on the test set and returns various classification metrics.
    
    Args:
        model: Trained scikit-learn compatible model.
        X_test: Test features.
        y_test: Test labels.
        
    Returns:
        dict: A dictionary containing performance metrics.
    """
    if model is None:
        return {}
        
    y_pred = model.predict(X_test)
    
    # Check if model supports predict_proba for ROC-AUC
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = y_pred
        
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }
    
    try:
        metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
    except ValueError:
        # This typically happens if the test set only contains one class
        metrics['roc_auc'] = np.nan
        
    metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
    
    return metrics
