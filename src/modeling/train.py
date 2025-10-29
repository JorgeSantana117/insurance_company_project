# src/modeling/train.py
"""
Model training, splitting, and evaluation functions.
"""
import pandas as pd
import numpy as np
import time

# MLflow
import mlflow
import mlflow.sklearn

# scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

# XGBoost
from xgboost import XGBClassifier

# --- 1. Data Splitting ---

def split_data(X: pd.DataFrame, y: pd.Series, random_state: int) -> tuple:
    """
    Splits data into train (60%), validation (20%), and test (20%) sets.
    """
    # Split into Train+Valid (80%) and Test (20%)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=random_state
    )
    
    # Split Train+Valid into Train (60%) and Valid (20%)
    # test_size=0.25 means 25% of 80% = 20% of total
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=random_state
    )
    
    print(f"Data split complete:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Valid: {X_valid.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test

# --- 2. Custom Metrics ---

def _specificity_score(y_true, y_pred):
    """Calculates Specificity (True Negative Rate)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def _gmean_at_threshold(y_true, y_proba, thr):
    """Calculates G-Mean for a given threshold."""
    y_hat = (y_proba >= thr).astype(int)
    rec = recall_score(y_true, y_hat)      # Sensitivity
    spc = _specificity_score(y_true, y_hat) # Specificity
    return np.sqrt(rec * spc), rec, spc

def _find_best_threshold(y_true, p_valid):
    """Finds the optimal threshold on validation data using G-Mean."""
    thresholds = np.linspace(0.01, 0.99, 99)
    gmeans = []
    for t in thresholds:
        g, r, s = _gmean_at_threshold(y_true, p_valid, t)
        gmeans.append(g)
    
    best_idx = int(np.argmax(gmeans))
    best_thr = float(thresholds[best_idx])
    best_gmean = float(gmeans[best_idx])
    
    return best_thr, best_gmean

# --- 3. Model Training: Logistic Regression ---

def train_logistic_regression(X_train, y_train, X_valid, y_valid, X_test, y_test, 
                              preprocessor, random_state, mlflow_run_name: str) -> dict:
    """
    Trains and evaluates a Logistic Regression baseline model with MLflow tracking.
    """
    print(f"\n--- Training Model 1: {mlflow_run_name} ---")
    
    # Start MLflow run
    with mlflow.start_run(run_name=mlflow_run_name) as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        
        start_time = time.time()
        
        # Define model parameters
        model_params = {
            "max_iter": 2000,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "random_state": random_state,
            "n_jobs": -1
        }
        clf = LogisticRegression(**model_params)
        
        # Create full pipeline
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", clf)
        ])
        
        # Train
        pipe.fit(X_train, y_train)
        print(f"Training complete in {time.time() - start_time:.2f}s")
        
        # --- Evaluation ---
        p_valid = pipe.predict_proba(X_valid)[:, 1]
        best_thr, gmean_valid = _find_best_threshold(y_valid, p_valid)
        
        p_test = pipe.predict_proba(X_test)[:, 1]
        yhat_test = (p_test >= best_thr).astype(int)
        gmean_test, rec_test, spc_test = _gmean_at_threshold(y_test, p_test, best_thr)
        
        # Compile results
        results = {
            "model": "Logistic Regression",
            "best_threshold_gmean": best_thr,
            "valid_gmean": gmean_valid,
            "test_pr_auc": average_precision_score(y_test, p_test),
            "test_roc_auc": roc_auc_score(y_test, p_test),
            "test_gmean": gmean_test,
            "test_recall_TPR": rec_test,
            "test_specificity_TNR": spc_test,
            "test_f1_score": f1_score(y_test, yhat_test),
            "test_confusion_matrix": confusion_matrix(y_test, yhat_test, labels=[0, 1]).tolist(),
            "mlflow_run_id": run_id
        }
        
        # --- MLflow Logging ---
        # Autolog handles params and model artifact.
        # We manually log our superior, threshold-tuned metrics.
        metrics_to_log = {k: v for k, v in results.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(metrics_to_log)
        
        results["mlflow_model_uri"] = mlflow.get_artifact_uri("model")
        print("--- Logistic Regression Evaluation (on Test Set) ---")
        for k, v in results.items():
            if k == "test_confusion_matrix":
                print(f"  {k}: \n{np.array(v)}")
            elif isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
                
        return results

# --- 4. Model Training: XGBoost ---

def train_xgboost(X_train, y_train, X_valid, y_valid, X_test, y_test, 
                  preprocessor, random_state, mlflow_run_name: str) -> dict:
    """
    Trains and evaluates an XGBoost model using RandomizedSearchCV with MLflow tracking.
    """
    print(f"\n--- Training Model 2: {mlflow_run_name} ---")
    
    # Start MLflow run
    with mlflow.start_run(run_name=mlflow_run_name) as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")
        
        start_time = time.time()
        
        # Calculate scale_pos_weight for imbalanced dataset
        pos = int(y_train.sum())
        neg = int(len(y_train) - pos)
        ratio = neg / max(1, pos)

        xgb = XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            enable_categorical=False
        )
        
        pipe_xgb = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", xgb)
        ])
        
        # Hyperparameter search space
        param_dist = {
            "model__n_estimators":      [300, 500, 800, 1000],
            "model__max_depth":         [4, 5, 6, 7],
            "model__learning_rate":     [0.01, 0.05, 0.1],
            "model__subsample":         [0.7, 0.8, 0.9, 1.0],
            "model__colsample_bytree":  [0.7, 0.8, 0.9, 1.0],
            "model__min_child_weight":  [1, 3, 5],
            "model__gamma":             [0, 0.1, 0.3, 0.5],
            "model__reg_alpha":         [0, 0.01, 0.1],
            "model__reg_lambda":        [0.1, 1.0, 5.0],
            "model__scale_pos_weight":  [1, ratio, ratio/2, np.sqrt(ratio), 10, 15]
        }
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        
        # n_iter=10 for speed. Increase for a more thorough search.
        search = RandomizedSearchCV(
            estimator=pipe_xgb,
            param_distributions=param_dist,
            n_iter=10, 
            scoring="average_precision",
            n_jobs=-1,
            cv=cv,
            refit=True,
            verbose=1,
            random_state=random_state
        )
        
        search.fit(X_train, y_train)
        
        print(f"\nRandomizedSearchCV complete in {time.time() - start_time:.2f}s")
        print(f"Best AP (cv): {search.best_score_:.4f}")
        
        best_pipe = search.best_estimator_
        
        # --- Evaluation ---
        p_valid = best_pipe.predict_proba(X_valid)[:, 1]
        best_thr, gmean_valid = _find_best_threshold(y_valid, p_valid)
        
        p_test = best_pipe.predict_proba(X_test)[:, 1]
        yhat_test = (p_test >= best_thr).astype(int)
        gmean_test, rec_test, spc_test = _gmean_at_threshold(y_test, p_test, best_thr)
        
        # Compile results
        results = {
            "model": "XGBoost",
            "best_threshold_gmean": best_thr,
            "valid_gmean": gmean_valid,
            "test_pr_auc": average_precision_score(y_test, p_test),
            "test_roc_auc": roc_auc_score(y_test, p_test),
            "test_gmean": gmean_test,
            "test_recall_TPR": rec_test,
            "test_specificity_TNR": spc_test,
            "test_f1_score": f1_score(y_test, yhat_test),
            "test_confusion_matrix": confusion_matrix(y_test, yhat_test, labels=[0, 1]).tolist(),
            "best_cv_avg_precision": search.best_score_,
            "mlflow_run_id": run_id
        }

        # --- MLflow Logging ---
        # Autolog handles params and model artifact.
        # We manually log our superior, threshold-tuned metrics.
        metrics_to_log = {k: v for k, v in results.items() if isinstance(v, (int, float))}
        mlflow.log_metrics(metrics_to_log)
        
        results["mlflow_model_uri"] = mlflow.get_artifact_uri("model")
        
        print("\n--- XGBoost Evaluation (on Test Set) ---")
        for k, v in results.items():
            if k == "test_confusion_matrix":
                print(f"  {k}: \n{np.array(v)}")
            elif k == "best_params":
                continue 
            elif isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
                
        return results