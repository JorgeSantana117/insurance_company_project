# src/config.py
"""
Configuration file for the MLOps project.
Stores file paths, column lists, and model parameters.
"""

from pathlib import Path

# --- Project Root ---
# Assumes this config.py is in 'src/', and the project root is its parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- File Paths ---
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "insurance_company_modified.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "insurance_company_cleaned_modified.csv"
MODEL_REGISTRY_PATH = PROJECT_ROOT / "models"

# --- MLflow Settings ---
MLFLOW_EXPERIMENT_NAME = "Insurance_Company_Benchmark"

# --- Global Settings ---
RANDOM_STATE = 42
RARE_THRESHOLD = 0.01  # 1%
PRUNING_THRESHOLD = 0.85 # Correlation threshold for pruning

# --- Original Short Column Names (85 features + 1 target) ---
# Used for loading and cleaning the raw data (which has no headers)
COLS = [
    "MOSTYPE","MAANTHUI","MGEMOMV","MGEMLEEF","MOSHOOFD","MGODRK","MGODPR","MGODOV","MGODGE",
    "MRELGE","MRELSA","MRELOV","MFALLEEN","MFGEKIND","MFWEKIND","MOPLHOOG","MOPLMIDD","MOPLLAAG",
    "MBERHOOG","MBERZELF","MBERBOER","MBERMIDD","MBERARBG","MBERARBO","MSKA","MSKB1","MSKB2","MSKC",
    "MSKD","MHHUUR","MHKOOP","MAUT1","MAUT2","MAUT0","MZFONDS","MZPART","MINKM30","MINK3045","MINK4575",
    "MINK7512","MINK123M","MINKGEM","MKOOPKLA","PWAPART","PWABEDR","PWALAND","PPERSAUT","PBESAUT","PMOTSCO",
    "PVRAAUT","PAANHANG","PTRACTOR","PWERKT","PBROM","PLEVEN","PPERSONG","PGEZONG","PWAOREG","PBRAND",
    "PZEILPL","PPLEZIER","PFIETS","PINBOED","PBYSTAND","AWAPART","AWABEDR","AWALAND","APERSAUT","ABESAUT",
    "AMOTSCO","AVRAAUT","AAANHANG","ATRACTOR","AWERKT","ABROM","ALEVEN","APERSONG","AGEZONG","AWAOREG",
    "ABRAND","AZEILPL","APLEZIER","AFIETS","AINBOED","ABYSTAND","CARAVAN"
]

# --- Expected Data Ranges (from dictionary.txt) ---
def build_expected():
    """Builds the dictionary of expected ranges for data cleaning."""
    expected = {
        "MOSTYPE": (1, 41),
        "MAANTHUI": (1, 10),
        "MGEMOMV": (1, 6),
        "MGEMLEEF": (1, 6),
        "MOSHOOFD": (1, 10),
        "CARAVAN": (0, 1),
    }
    fields_0_9 = [
        "MGODRK","MGODPR","MGODOV","MGODGE","MRELGE","MRELSA","MRELOV","MFALLEEN","MFGEKIND","MFWEKIND",
        "MOPLHOOG","MOPLMIDD","MOPLLAAG","MBERHOOG","MBERZELF","MBERBOER","MBERMIDD","MBERARBG","MBERARBO",
        "MSKA","MSKB1","MSKB2","MSKC","MSKD","MHHUUR","MHKOOP","MAUT1","MAUT2","MAUT0","MZFONDS","MZPART",
        "MINKM30","MINK3045","MINK4575","MINK7512","MINK123M","MINKGEM","MKOOPKLA",
        "PWAPART","PWABEDR","PWALAND","PPERSAUT","PBESAUT","PMOTSCO","PVRAAUT","PAANHANG","PTRACTOR","PWERKT",
        "PBROM","PLEVEN","PPERSONG","PGEZONG","PWAOREG","PBRAND","PZEILPL","PPLEZIER","PFIETS","PINBOED","PBYSTAND",
    ]
    for f in fields_0_9:
        expected[f] = (0, 9)
    
    a_fields = [
        "AWAPART","AWABEDR","AWALAND","APERSAUT","ABESAUT","AMOTSCO","AVRAAUT","AAANHANG",
        "ATRACTOR","AWERKT","ABROM","ALEVEN","APERSONG","AGEZONG","AWAOREG","ABRAND","AZEILPL",
        "APLEZIER","AFIETS","AINBOED","ABYSTAND"
    ]
    for a in a_fields:
        expected[a] = (1, 12)
    return expected

EXPECTED_RANGES = build_expected()

# --- Descriptive Column Names (85 features + 1 target) ---
# Used to rename columns after cleaning for better readability
LONG_NAMES = [
    "customer_subtype","num_houses","avg_household_size","avg_age_band","customer_main_type",
    "pct_roman_catholic","pct_protestant","pct_other_religion","pct_no_religion",
    "pct_married","pct_living_together","pct_other_relation","pct_singles","pct_household_no_kids",
    "pct_household_with_kids","pct_high_education","pct_medium_education","pct_low_education",
    "pct_high_status","pct_entrepreneur","pct_farmer","pct_middle_management","pct_skilled_labour",
    "pct_unskilled_labour","pct_social_class_a","pct_social_class_b1","pct_social_class_b2",
    "pct_social_class_c","pct_social_class_d","pct_rented_house","pct_home_owner","pct_one_car",
    "pct_two_cars","pct_no_car","pct_national_health_service","pct_private_health_insurance",
    "pct_income_lt_30k","pct_income_30k_45k","pct_income_45k_75k","pct_income_75k_122k",
    "pct_income_gt_123k","pct_avg_income","purchasing_power_class",
    "contr_private_third_party","contr_third_party_firms","contr_third_party_agriculture",
    "contr_car_policies","contr_delivery_van_policies","contr_motorcycle_policies","contr_lorry_policies",
    "contr_trailer_policies","contr_tractor_policies","contr_agri_machine_policies","contr_moped_policies",
    "contr_life_ins","contr_private_accident","contr_family_accident","contr_disability_ins",
    "contr_fire_policies","contr_surfboard_policies","contr_boat_policies","contr_bicycle_policies",
    "contr_property_ins","contr_social_security",
    "num_private_third_party","num_third_party_firms","num_third_party_agriculture","num_car_policies",
    "num_delivery_van_policies","num_motorcycle_policies","num_lorry_policies","num_trailer_policies",
    "num_tractor_policies","num_agri_machine_policies","num_moped_policies","num_life_ins",
    "num_private_accident","num_family_accident","num_disability_ins","num_fire_policies",
    "num_surfboard_policies","num_boat_policies","num_bicycle_policies","num_property_ins",
    "num_social_security",
    "target_caravan"
]

# --- Feature Type Lists (based on long names) ---
TARGET_COL = "target_caravan"

NOMINAL_COLS = [
    "customer_subtype",
    "customer_main_type"
]

ORDINAL_COLS = [
    "avg_age_band","pct_roman_catholic","pct_protestant","pct_other_religion","pct_no_religion",
    "pct_married","pct_living_together","pct_other_relation","pct_singles","pct_household_no_kids",
    "pct_household_with_kids","pct_high_education","pct_medium_education","pct_low_education",
    "pct_high_status","pct_entrepreneur","pct_farmer","pct_middle_management","pct_skilled_labour",
    "pct_unskilled_labour","pct_social_class_a","pct_social_class_b1","pct_social_class_b2",
    "pct_social_class_c","pct_social_class_d","pct_rented_house","pct_home_owner","pct_one_car",
    "pct_two_cars","pct_no_car","pct_national_health_service","pct_private_health_insurance",
    "pct_income_lt_30k","pct_income_30k_45k","pct_income_45k_75k","pct_income_75k_122k",
    "pct_income_gt_123k","pct_avg_income","purchasing_power_class",
    "contr_private_third_party","contr_third_party_firms","contr_third_party_agriculture",
    "contr_car_policies","contr_delivery_van_policies","contr_motorcycle_policies","contr_lorry_policies",
    "contr_trailer_policies","contr_tractor_policies","contr_agri_machine_policies","contr_moped_policies",
    "contr_life_ins","contr_private_accident","contr_family_accident","contr_disability_ins",
    "contr_fire_policies","contr_surfboard_policies","contr_boat_policies","contr_bicycle_policies",
    "contr_property_ins","contr_social_security"
]

NUMERIC_DISCRETE_COLS = [
    "num_houses","avg_household_size",
    "num_private_third_party","num_third_party_firms","num_third_party_agriculture","num_car_policies",
    "num_delivery_van_policies","num_motorcycle_policies","num_lorry_policies","num_trailer_policies",
    "num_tractor_policies","num_agri_machine_policies","num_moped_policies","num_life_ins",
    "num_private_accident","num_family_accident","num_disability_ins","num_fire_policies",
    "num_surfboard_policies","num_boat_policies","num_bicycle_policies","num_property_ins",
    "num_social_security"
]