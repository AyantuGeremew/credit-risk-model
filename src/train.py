import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

#-----------------------------------------------
# Data Preparation
#-------------------------------------------

def split_data(
    df,
    target_col="is_high_risk",
    test_size=0.2,
    random_state=42
):
    """
    Split dataset into training and testing sets.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset containing features and target.
    target_col : str
        Target variable column name.
    test_size : float
        Proportion of data to use for testing.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # preserves class distribution
    )

    return X_train, X_test, y_train, y_test

#-----------------------------------------------
# Model Selection and Training
#-------------------------------------------

def train_models(X_train, y_train, random_state=42):
    """
    Train multiple classification models.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    random_state : int
        Ensures reproducibility

    Returns
    -------
    dict
        Trained models
    """

    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        
        "Decision_Tree": DecisionTreeClassifier(random_state=random_state),
        
        "Random_Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state
        ),
        
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state,
            eval_metric="logloss"
        )
    }

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models

def evaluate_models(models, X_test, y_test):
    """
    Evaluate trained models on test data.
    """

    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)

        print(f"\n===== {name} =====")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        results[name] = accuracy_score(y_test, y_pred)

    return results

#-----------------------------------------------
# Hyperparameter Tuning
#-------------------------------------------

def tune_model_grid_search(model, param_grid, X_train, y_train, cv=5):
    """
    Perform hyperparameter tuning using Grid Search.

    Parameters
    ----------
    model : sklearn model
        Base model to tune
    param_grid : dict
        Hyperparameter grid
    X_train : pd.DataFrame
    y_train : pd.Series
    cv : int
        Cross-validation folds

    Returns
    -------
    best_model : trained model
    best_params : dict
    """

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_

def tune_model_random_search(model, param_dist, X_train, y_train, n_iter=20, cv=5, random_state=42):
    """
    Perform hyperparameter tuning using Random Search.

    Parameters
    ----------
    model : sklearn model
    param_dist : dict
        Hyperparameter distributions
    X_train : pd.DataFrame
    y_train : pd.Series
    n_iter : int
        Number of parameter settings sampled
    cv : int
        Cross-validation folds
    random_state : int
        Ensures reproducibility

    Returns
    -------
    best_model : trained model
    best_params : dict
    """

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="accuracy",
        random_state=random_state,
        n_jobs=-1
    )

    random_search.fit(X_train, y_train)

    return random_search.best_estimator_, random_search.best_params_

mlflow.set_experiment("credit_risk_model_experiments")

def evaluate_model(model, X_test, y_test):
    """
    Compute evaluation metrics.
    """

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    return metrics

def log_model_mlflow(model, model_name, params, metrics, X_train):
    """
    Log model parameters, metrics, and artifacts to MLflow.
    """

    with mlflow.start_run(run_name=model_name):

        # Log parameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            input_example=X_train.head(5)
        )

        print(f"{model_name} logged to MLflow")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def run_experiments(X_train, X_test, y_train, y_test):
    """
    Train multiple models and log them to MLflow.
    """

    models = {
        "Logistic_Regression": LogisticRegression(max_iter=1000),
        "Random_Forest": RandomForestClassifier(random_state=42)
    }

    results = {}

    for name, model in models.items():

        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        params = model.get_params()

        log_model_mlflow(
            model=model,
            model_name=name,
            params=params,
            metrics=metrics,
            X_train=X_train
        )

        results[name] = metrics

    return results

def register_best_model(run_id, model_name="best_credit_risk_model"):
    """
    Register best model in MLflow Model Registry.
    """

    model_uri = f"runs:/{run_id}/model"

    mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

def evaluate_classification_model(model, X_test, y_test):
    """
    Evaluate a classification model using key performance metrics.

    Parameters
    ----------
    model : trained ML model
        Fitted classification model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        True labels

    Returns
    -------
    dict
        Dictionary of evaluation metrics
    """

    # Predictions
    y_pred = model.predict(X_test)

    # Probabilities (needed for ROC-AUC)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    return metrics


