# src/api/service.py
import os
from typing import List

import mlflow
import pandas as pd

from .schemas import InsuranceRequest


# URI por defecto al modelo, basada en tu run_id del MLmodel
# Puedes sobrescribirla con la variable de entorno MODEL_URI
DEFAULT_MODEL_URI = "runs:/3ba2c4ed3a1f45cb998642e19ed4219a/model"


# Lista de columnas en el orden de la signature (por seguridad)
FEATURE_NAMES: List[str] = [
    "num_houses",
    "avg_household_size",
    "avg_age_band",
    "customer_main_type",
    "pct_roman_catholic",
    "pct_protestant",
    "pct_other_religion",
    "pct_no_religion",
    "pct_married",
    "pct_living_together",
    "pct_singles",
    "pct_household_no_kids",
    "pct_household_with_kids",
    "pct_high_education",
    "pct_medium_education",
    "pct_low_education",
    "pct_high_status",
    "pct_entrepreneur",
    "pct_farmer",
    "pct_middle_management",
    "pct_skilled_labour",
    "pct_unskilled_labour",
    "pct_social_class_a",
    "pct_social_class_b1",
    "pct_social_class_b2",
    "pct_social_class_c",
    "pct_social_class_d",
    "pct_home_owner",
    "pct_one_car",
    "pct_two_cars",
    "pct_no_car",
    "pct_private_health_insurance",
    "pct_income_lt_30k",
    "pct_income_30k_45k",
    "pct_income_45k_75k",
    "pct_income_75k_122k",
    "pct_income_gt_123k",
    "pct_avg_income",
    "purchasing_power_class",
    "contr_private_third_party",
    "contr_third_party_firms",
    "contr_third_party_agriculture",
    "contr_car_policies",
    "contr_delivery_van_policies",
    "contr_motorcycle_policies",
    "contr_lorry_policies",
    "contr_trailer_policies",
    "contr_tractor_policies",
    "contr_agri_machine_policies",
    "contr_moped_policies",
    "contr_life_ins",
    "contr_private_accident",
    "contr_family_accident",
    "contr_disability_ins",
    "contr_fire_policies",
    "contr_surfboard_policies",
    "contr_boat_policies",
    "contr_bicycle_policies",
    "contr_property_ins",
    "contr_social_security",
    "num_private_third_party",
    "num_third_party_firms",
    "num_third_party_agriculture",
    "num_car_policies",
    "num_delivery_van_policies",
    "num_motorcycle_policies",
    "num_lorry_policies",
    "num_trailer_policies",
    "num_tractor_policies",
    "num_agri_machine_policies",
    "num_moped_policies",
    "num_life_ins",
    "num_private_accident",
    "num_family_accident",
    "num_disability_ins",
    "num_fire_policies",
    "num_surfboard_policies",
    "num_boat_policies",
    "num_bicycle_policies",
    "num_property_ins",
    "num_social_security",
]


class InsuranceModelService:
    def __init__(self) -> None:
        # Permite cambiar el modelo con una variable de entorno
        self.model_uri = os.getenv("MODEL_URI", DEFAULT_MODEL_URI)

        # Carga el modelo sklearn desde MLflow
        # (usa mlflow.sklearn para poder llamar predict_proba)
        self.model = mlflow.sklearn.load_model(self.model_uri)

    def predict(self, request: InsuranceRequest, threshold: float = 0.5) -> dict:
        """
        Toma un InsuranceRequest, lo convierte a DataFrame ordenando columnas
        según la signature y devuelve clase predicha + probabilidad (si existe).
        """
        # Pydantic -> dict
        # features = request.dict()  # también válido
        features = request.model_dump()

        # Ordenar columnas de acuerdo a FEATURE_NAMES
        row = {name: features[name] for name in FEATURE_NAMES}
        X = pd.DataFrame([row])

        # Predicción de clase (lo que define tu MLmodel: salida int64)
        y_pred = self.model.predict(X)
        predicted_class = int(y_pred[0])

        # Probabilidad, si el modelo lo soporta
        proba = None
        if hasattr(self.model, "predict_proba"):
            proba_vals = self.model.predict_proba(X)
            # Asumimos prob de clase 1 en la segunda columna
            proba = float(proba_vals[0, 1])

        return {
            "predicted_class": predicted_class,
            "probability": proba,
            "mlflow_model_uri": self.model_uri,
        }


# Instancia global reutilizable por FastAPI
service = InsuranceModelService()
