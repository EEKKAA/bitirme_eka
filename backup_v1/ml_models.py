"""
Machine learning models for bankruptcy prediction.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from config import RANDOM_STATE

def train_logistic(X_train, y_train):
    """Trains a Logistic Regression model with class weight balancing."""
    model = LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced', max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Trains a Random Forest classifier with class weight balancing."""
    model = RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced', n_estimators=100)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    """Trains an XGBoost classifier, handling class imbalance."""
    if not XGB_AVAILABLE:
        print("XGBoost is not installed. Skipping XGBoost model.")
        return None
        
    # Calculate scale_pos_weight to handle class imbalance
    num_neg = sum(y_train == 0)
    num_pos = sum(y_train == 1)
    pos_weight = num_neg / max(1, num_pos)
    
    model = xgb.XGBClassifier(
        random_state=RANDOM_STATE, 
        scale_pos_weight=pos_weight, 
        use_label_encoder=False, 
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model
