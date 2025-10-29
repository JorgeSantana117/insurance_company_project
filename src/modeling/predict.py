# src/modeling/predict.py
"""
Functions for loading a trained model and making predictions.
"""
import pandas as pd
from sklearn.pipeline import Pipeline
import joblib # For loading models

def load_model(path: str) -> Pipeline:
    """Loads a serialized model pipeline."""
    model = joblib.load(path)
    return model

def make_predictions(input_data: pd.DataFrame, model: Pipeline) -> pd.Series:
    """Makes predictions on new data."""
    predictions = model.predict(input_data)
    return predictions

def make_probabilities(input_data: pd.DataFrame, model: Pipeline) -> pd.Series:
    """Gets class probabilities on new data."""
    probabilities = model.predict_proba(input_data)[:, 1]
    return probabilities