# src/api/schemas.py
from typing import Optional

from pydantic import BaseModel, Field


class InsuranceRequest(BaseModel):
    """
    Base del modelo de entrada.
    """
    num_houses: int = Field(..., description="Número de casas")
    avg_household_size: int
    avg_age_band: int
    customer_main_type: int
    pct_roman_catholic: int
    pct_protestant: int
    pct_other_religion: int
    pct_no_religion: int
    pct_married: int
    pct_living_together: int
    pct_singles: int
    pct_household_no_kids: int
    pct_household_with_kids: int
    pct_high_education: int
    pct_medium_education: int
    pct_low_education: int
    pct_high_status: int
    pct_entrepreneur: int
    pct_farmer: int
    pct_middle_management: int
    pct_skilled_labour: int
    pct_unskilled_labour: int
    pct_social_class_a: int
    pct_social_class_b1: int
    pct_social_class_b2: int
    pct_social_class_c: int
    pct_social_class_d: int
    pct_home_owner: int
    pct_one_car: int
    pct_two_cars: int
    pct_no_car: int
    pct_private_health_insurance: int
    pct_income_lt_30k: int
    pct_income_30k_45k: int
    pct_income_45k_75k: int
    pct_income_75k_122k: int
    pct_income_gt_123k: int
    pct_avg_income: int
    purchasing_power_class: int
    contr_private_third_party: int
    contr_third_party_firms: int
    contr_third_party_agriculture: int
    contr_car_policies: int
    contr_delivery_van_policies: int
    contr_motorcycle_policies: int
    contr_lorry_policies: int
    contr_trailer_policies: int
    contr_tractor_policies: int
    contr_agri_machine_policies: int
    contr_moped_policies: int
    contr_life_ins: int
    contr_private_accident: int
    contr_family_accident: int
    contr_disability_ins: int
    contr_fire_policies: int
    contr_surfboard_policies: int
    contr_boat_policies: int
    contr_bicycle_policies: int
    contr_property_ins: int
    contr_social_security: int
    num_private_third_party: int
    num_third_party_firms: int
    num_third_party_agriculture: int
    num_car_policies: int
    num_delivery_van_policies: int
    num_motorcycle_policies: int
    num_lorry_policies: int
    num_trailer_policies: int
    num_tractor_policies: int
    num_agri_machine_policies: int
    num_moped_policies: int
    num_life_ins: int
    num_private_accident: int
    num_family_accident: int
    num_disability_ins: int
    num_fire_policies: int
    num_surfboard_policies: int
    num_boat_policies: int
    num_bicycle_policies: int
    num_property_ins: int
    num_social_security: int


class InsuranceResponse(BaseModel):
    """
    Respuesta estándar del servicio.
    """
    predicted_class: int = Field(
        ...,
        description="Clase predicha por el modelo (0 = no CARAVAN, 1 = CARAVAN).",
    )
    probability: Optional[float] = Field(
        None,
        description="Probabilidad estimada de CARAVAN=1 (si el modelo tiene predict_proba).",
    )
    mlflow_model_uri: str = Field(
        ...,
        description=(
            "URI o ruta del modelo en MLflow (p.ej. runs:/<run_id>/model "
            "o ruta local al directorio con MLmodel)."
        ),
    )
