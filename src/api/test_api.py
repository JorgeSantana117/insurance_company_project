import pandas as pd
import requests

API_URL = "http://localhost:8000/predict"
CSV_PATH = "test_api.csv"  # ajusta la ruta si está en otra carpeta

# Lista de FEATURES que espera tu servicio (mismo orden que en service.py)
FEATURE_NAMES = [
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

# Mapeo: campo que espera la API  ->  columna del CSV
FIELD_TO_CSV_COL = {
    "num_houses": "MAANTHUI",
    "avg_household_size": "MGEMOMV",
    "avg_age_band": "MGEMLEEF",
    "customer_main_type": "MOSHOOFD",
    "pct_roman_catholic": "MGODRK",
    "pct_protestant": "MGODPR",
    "pct_other_religion": "MGODOV",
    "pct_no_religion": "MGODGE",
    "pct_married": "MRELGE",
    "pct_living_together": "MRELSA",
    "pct_singles": "MRELOV",
    "pct_household_no_kids": "MFALLEEN",
    "pct_household_with_kids": "MFGEKIND",
    "pct_high_education": "MOPLHOOG",
    "pct_medium_education": "MOPLMIDD",
    "pct_low_education": "MOPLLAAG",
    "pct_high_status": "MBERHOOG",
    "pct_entrepreneur": "MBERZELF",
    "pct_farmer": "MBERBOER",
    "pct_middle_management": "MBERMIDD",
    "pct_skilled_labour": "MBERARBG",
    "pct_unskilled_labour": "MBERARBO",
    "pct_social_class_a": "MSKA",
    "pct_social_class_b1": "MSKB1",
    "pct_social_class_b2": "MSKB2",
    "pct_social_class_c": "MSKC",
    "pct_social_class_d": "MSKD",
    "pct_home_owner": "MHKOOP",
    "pct_one_car": "MAUT1",
    "pct_two_cars": "MAUT2",
    "pct_no_car": "MAUT0",
    "pct_private_health_insurance": "MZPART",
    "pct_income_lt_30k": "MINKM30",
    "pct_income_30k_45k": "MINK3045",
    "pct_income_45k_75k": "MINK4575",
    "pct_income_75k_122k": "MINK7512",
    "pct_income_gt_123k": "MINK123M",
    "pct_avg_income": "MINKGEM",
    # 'purchasing_power_class' lo dejamos calculado a partir de MINKGEM o 0
}


def build_payload_from_row(row: pd.Series) -> dict:
    """
    Construye el JSON que espera la API a partir de una fila del CSV.
    Para los campos que no estén mapeados (contr_* y num_*) se pone 0.
    """
    payload = {}

    for field in FEATURE_NAMES:
        if field in FIELD_TO_CSV_COL:
            col = FIELD_TO_CSV_COL[field]
            val = row.get(col, 0)
            # NaN -> 0
            if pd.isna(val):
                val = 0
            payload[field] = int(val)
        elif field == "purchasing_power_class":
            # Ejemplo: usar la renta media como proxy
            val = row.get("MINKGEM", 0)
            if pd.isna(val):
                val = 0
            payload[field] = int(val)
        else:
            # Para los contr_* y num_* que no tenemos 1:1 en el CSV
            payload[field] = 0

    return payload


def main():
    # 1) Leer el CSV
    df = pd.read_csv(CSV_PATH)

    # 2) Tomar un ejemplo: un cliente con CARAVAN = 1 (si existe)
    if "CARAVAN" in df.columns and (df["CARAVAN"] == 1).any():
        row = df[df["CARAVAN"] == 1].sample(1, random_state=42).iloc[0]
        print("Usando un ejemplo con CARAVAN = 1")
    else:
        row = df.sample(1, random_state=42).iloc[0]
        print("Usando un ejemplo aleatorio (no se encontró CARAVAN = 1)")

    # 3) Construir el payload
    payload = build_payload_from_row(row)

    print("\nPayload que se enviará a la API:")
    for k, v in list(payload.items())[:10]:
        print(f"  {k}: {v}")
    print("  ...")

    # 4) Hacer la petición POST
    resp = requests.post(API_URL, json=payload)

    print("\nStatus code:", resp.status_code)
    print("Respuesta JSON:")
    try:
        print(resp.json())
    except Exception:
        print(resp.text)


if __name__ == "__main__":
    main()
