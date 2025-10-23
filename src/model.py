print("Entrenando y evaluando modelo...")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder, OneHotEncoder, StandardScaler, RobustScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score, recall_score, confusion_matrix
)
from sklearn.compose import make_column_selector
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
# 1) Carga el dataset limpio del paso anterior
df_copy = pd.read_csv('df_copy_eda.csv')


# ------------------------------------------------------------------
# 1) Asignar nombres "cortos" originales en orden estándar (1..86)
#    y nombres descriptivos para mayor legibilidad.
#    El orden es el de TICDATA/COIL2000 (85 features + target).
# ------------------------------------------------------------------

short_names = [
    # 1..43: sociodemográfico (zip code derived)
    "MOSTYPE","MAANTHUI","MGEMOMV","MGEMLEEF","MOSHOOFD","MGODRK","MGODPR","MGODOV","MGODGE",
    "MRELGE","MRELSA","MRELOV","MFALLEEN","MFGEKIND","MFWEKIND","MOPLHOOG","MOPLMIDD","MOPLLAAG",
    "MBERHOOG","MBERZELF","MBERBOER","MBERMIDD","MBERARBG","MBERARBO","MSKA","MSKB1","MSKB2","MSKC",
    "MSKD","MHHUUR","MHKOOP","MAUT1","MAUT2","MAUT0","MZFONDS","MZPART","MINKM30","MINK3045",
    "MINK4575","MINK7512","MINK123M","MINKGEM","MKOOPKLA",
    # 44..64: P* contribuciones (rangos monetarios)
    "PWAPART","PWABEDR","PWALAND","PPERSAUT","PBESAUT","PMOTSCO","PVRAAUT","PAANHANG","PTRACTOR",
    "PWERKT","PBROM","PLEVEN","PPERSONG","PGEZONG","PWAOREG","PBRAND","PZEILPL","PPLEZIER","PFIETS",
    "PINBOED","PBYSTAND",
    # 65..85: A* conteos de pólizas
    "AWAPART","AWABEDR","AWALAND","APERSAUT","ABESAUT","AMOTSCO","AVRAAUT","AAANHANG","ATRACTOR",
    "AWERKT","ABROM","ALEVEN","APERSONG","AGEZONG","AWAOREG","ABRAND","AZEILPL","APLEZIER","AFIETS",
    "AINBOED","ABYSTAND",
    # 86: target
    "CARAVAN"
]

# Nombres descriptivos (alineados 1 a 1 con short_names)
long_names = [
    # 1..43
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
    # 44..64 (rangos monetarios, ordinales)
    "contr_private_third_party","contr_third_party_firms","contr_third_party_agriculture",
    "contr_car_policies","contr_delivery_van_policies","contr_motorcycle_policies","contr_lorry_policies",
    "contr_trailer_policies","contr_tractor_policies","contr_agri_machine_policies","contr_moped_policies",
    "contr_life_ins","contr_private_accident","contr_family_accident","contr_disability_ins",
    "contr_fire_policies","contr_surfboard_policies","contr_boat_policies","contr_bicycle_policies",
    "contr_property_ins","contr_social_security",
    # 65..85 (conteos)
    "num_private_third_party","num_third_party_firms","num_third_party_agriculture","num_car_policies",
    "num_delivery_van_policies","num_motorcycle_policies","num_lorry_policies","num_trailer_policies",
    "num_tractor_policies","num_agri_machine_policies","num_moped_policies","num_life_ins",
    "num_private_accident","num_family_accident","num_disability_ins","num_fire_policies",
    "num_surfboard_policies","num_boat_policies","num_bicycle_policies","num_property_ins",
    "num_social_security",
    # target
    "target_caravan"
]

# Validación rápida del DataFrame de entrada:
assert df_copy.shape[1] == 86, "Se esperaban 86 columnas (85 features + target)."
df_copy = df_copy.copy()
df_copy.columns = long_names  # asigna nombres descriptivos

# ------------------------------------------------------------------
# 2) Definir listas de columnas por tipo semántico
#    - Nominales: códigos sin orden (tipologías)
#    - Ordinales: rangos crecientes (0..9 o bandas ascendentes)
#    - Numéricas discretas: conteos/magnitudes
# ------------------------------------------------------------------

nominal_cols = [
    "customer_subtype",      # MOSTYPE
    "customer_main_type"     # MOSHOOFD
]

# M* (porcentajes/bandas) + MGEMLEEF (edad por bandas) + contribuciones P* (rangos monetarios)
ordinal_cols = [
    # M* (excepto MAANTHUI, MGEMOMV, que son numéricas)
    "avg_age_band","pct_roman_catholic","pct_protestant","pct_other_religion","pct_no_religion",
    "pct_married","pct_living_together","pct_other_relation","pct_singles","pct_household_no_kids",
    "pct_household_with_kids","pct_high_education","pct_medium_education","pct_low_education",
    "pct_high_status","pct_entrepreneur","pct_farmer","pct_middle_management","pct_skilled_labour",
    "pct_unskilled_labour","pct_social_class_a","pct_social_class_b1","pct_social_class_b2",
    "pct_social_class_c","pct_social_class_d","pct_rented_house","pct_home_owner","pct_one_car",
    "pct_two_cars","pct_no_car","pct_national_health_service","pct_private_health_insurance",
    "pct_income_lt_30k","pct_income_30k_45k","pct_income_45k_75k","pct_income_75k_122k",
    "pct_income_gt_123k","pct_avg_income","purchasing_power_class",
    # P* (rangos monetarios)
    "contr_private_third_party","contr_third_party_firms","contr_third_party_agriculture",
    "contr_car_policies","contr_delivery_van_policies","contr_motorcycle_policies","contr_lorry_policies",
    "contr_trailer_policies","contr_tractor_policies","contr_agri_machine_policies","contr_moped_policies",
    "contr_life_ins","contr_private_accident","contr_family_accident","contr_disability_ins",
    "contr_fire_policies","contr_surfboard_policies","contr_boat_policies","contr_bicycle_policies",
    "contr_property_ins","contr_social_security"
]

# Numéricas discretas (conteos): MAANTHUI, MGEMOMV y todas las A* (incl. ABYSTAND al final)
numeric_discrete_cols = [
    "num_houses","avg_household_size",
    "num_private_third_party","num_third_party_firms","num_third_party_agriculture","num_car_policies",
    "num_delivery_van_policies","num_motorcycle_policies","num_lorry_policies","num_trailer_policies",
    "num_tractor_policies","num_agri_machine_policies","num_moped_policies","num_life_ins",
    "num_private_accident","num_family_accident","num_disability_ins","num_fire_policies",
    "num_surfboard_policies","num_boat_policies","num_bicycle_policies","num_property_ins",
    "num_social_security"
]



# ------------------------------------------------------------------
# 3) Eliminar variables numéricas altamente correlacionadas
#    - Umbral: se define un límite absoluto de correlación (thr=0.95)
#    - Estrategia: se elimina la más redundante según suma de correlaciones
#    - Excluye: la variable objetivo no se considera en el análisis
#    - Salida: DataFrame depurado, lista de columnas eliminadas, resumen
# ------------------------------------------------------------------

def prune_by_correlation(df_copy, target_col="target_caravan", thr=0.95, strategy="sum", random_state=42, verbose=True):
    """
    Remove redundant features by absolute correlation threshold.
    - df: DataFrame
    - target_col: name of target to exclude from correlation
    - thr: absolute correlation threshold above which we consider two features redundant
    - strategy: how to decide which one to drop in a high-correlation pair
        * "sum": drop the one with larger total correlation to others (more redundant globally)
        * "random": drop one at random (reproducible via random_state)
    Returns: pruned_df, dropped_columns, stats_dict
    """
    rng = np.random.RandomState(random_state)

    feat_cols = [c for c in df_copy.columns if c != target_col and pd.api.types.is_numeric_dtype(df_copy[c])]
    if len(feat_cols) < 2:
        return df_copy.copy(), [], {"reason": "not_enough_numeric", "n_features_original": len(feat_cols)}

    corr = df_copy[feat_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = set()
    kept = set(feat_cols)

    # Greedy elimination
    for col in upper.columns:
        # candidates highly correlated with col
        high_pairs = upper[col][upper[col] > thr].dropna()
        for row, val in high_pairs.items():
            if row in kept and col in kept:
                if strategy == "sum":
                    s_row = corr[row].sum()
                    s_col = corr[col].sum()
                    drop = row if s_row >= s_col else col
                else:  # random
                    drop = rng.choice([row, col])
                kept.discard(drop)
                to_drop.add(drop)

    pruned_df = df_copy.drop(columns=list(to_drop), errors="ignore")
    stats = {
        "n_features_original": len(feat_cols),
        "threshold": thr,
        "n_dropped": len(to_drop),
        "n_features_after_pruning": len(feat_cols) - len(to_drop)
    }
    if verbose:
        print("Pruning stats:", stats)
    return pruned_df, sorted(list(to_drop)), stats

# Run pruning
df_pruned, dropped_cols, stats_corr = prune_by_correlation(df_copy, target_col="CARAVAN", thr=0.95, strategy="sum", verbose=True)
print("Dropped (sample):", dropped_cols[:20])
print("Original shape:", df_copy.shape, "-> Pruned shape:", df_pruned.shape)




# ------------------------------------------------------------------
# 4) Definir transformadores por tipo
#    - Nominales -> OneHotEncoder
#    - Ordinales -> mantener como enteros ordenados + StandardScaler
#    - Numéricas discretas -> log1p (opcional) + RobustScaler (resistente a colas/outliers)
# ------------------------------------------------------------------

te = OneHotEncoder(handle_unknown="ignore", sparse_output=False) # codificamos las variables categóricas nominales.

ordinal_transformer = StandardScaler()  # ya están codificadas 0..9 o en bandas crecientes por lo que solo estandarizaremos.

numeric_discrete_transformer = Pipeline(steps=[
    ("log1p", FunctionTransformer(np.log1p, validate=False)),  # estabiliza sesgos si hay muchos ceros/colas
    ("scaler", RobustScaler())
])

# ------------------------------------------------------------------
# 5) Definir ColumnTransformer
# ------------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("nom", te, nominal_cols),
        ("ord", ordinal_transformer, ordinal_cols),
        ("num", numeric_discrete_transformer, numeric_discrete_cols),
    ],
    remainder="drop"
)


# ------------------------------------------------------------
# 6) Métricas y selección de umbral por G-Mean en VALID
# ------------------------------------------------------------
def specificity_score(y_true, y_pred):
    # TN / (TN + FP)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0

def gmean_at_threshold(y_true, y_proba, thr):
    y_hat = (y_proba >= thr).astype(int)
    rec = recall_score(y_true, y_hat)             # Sensitivity
    spc = specificity_score(y_true, y_hat)        # Specificity
    return np.sqrt(rec * spc), rec, spc

# ------------------------------------------------------------
# 7) Partición: Train / Valid / Test (estratificada)
#    - Train: 60% del total
#    - Test: 20% del total
#    - Valid: 20% del total
# ------------------------------------------------------------

# Separar variables dependientes e independiente
y = df_copy["target_caravan"].astype(int)
X = df_copy.drop(columns=["target_caravan"])

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42
)

# ------------------------------------------------------------
# 8) Pipeline final con Regresión Logística
#    - class_weight='balanced' para desbalance
# ------------------------------------------------------------
clf = LogisticRegression(
    max_iter=2000,
    class_weight="balanced", # dataset muy desbalanceado
    solver="lbfgs",
    n_jobs=-1
)

pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", clf)
])

# Entrenar
pipe.fit(X_train, y_train)

# Probabilidades en valid y test
p_valid = pipe.predict_proba(X_valid)[:, 1]
p_test  = pipe.predict_proba(X_test)[:, 1]

# Métricas umbral-independientes en VALID
ap_valid  = average_precision_score(y_valid, p_valid)
roc_valid = roc_auc_score(y_valid, p_valid)

# Buscar umbral óptimo por G-Mean
thresholds = np.linspace(0.01, 0.99, 99)
gmeans, recs, spcs = [], [], []
for t in thresholds:
    g, r, s = gmean_at_threshold(y_valid, p_valid, t)
    gmeans.append(g); recs.append(r); spcs.append(s)

best_idx = int(np.argmax(gmeans))
best_thr = float(thresholds[best_idx])
best_gmean_valid = float(gmeans[best_idx])

# ------------------------------------------------------------
# 9) Evaluación
# ------------------------------------------------------------
ap_test  = average_precision_score(y_test, p_test)
roc_test = roc_auc_score(y_test, p_test)

yhat_test = (p_test >= best_thr).astype(int)
gmean_test, rec_test, spc_test = gmean_at_threshold(y_test, p_test, best_thr)

yhat_valid = (p_valid >= best_thr).astype(int)
f1_valid = f1_score(y_valid, yhat_valid)
f1_test  = f1_score(y_test, yhat_test)

print("=== VALIDACIÓN (para selección de umbral) ===")
print(f"Average Precision (PR-AUC): {ap_valid:0.4f}")
print(f"ROC-AUC:                   {roc_valid:0.4f}")
print(f"Best threshold (G-Mean):   {best_thr:0.3f}")
print(f"G-Mean @best_thr:          {best_gmean_valid:0.4f}")
print(f"F1-score @best_thr:        {f1_valid:0.4f}")

print("\n=== PRUEBA (umbral fijado en valid) ===")
print(f"Average Precision (PR-AUC): {ap_test:0.4f}")
print(f"ROC-AUC:                   {roc_test:0.4f}")
print(f"G-Mean @thr={best_thr:0.3f}: {gmean_test:0.4f}")
print(f"  - Recall (TPR):          {rec_test:0.4f}")
print(f"  - Specificity (TNR):     {spc_test:0.4f}")
print(f"F1-score:                  {f1_test:0.4f}")

# Matriz de confusión en test
print("\nConfusion Matrix (Test)")
print(confusion_matrix(y_test, yhat_test, labels=[0,1]))


# ------------------------------------------------------------
# 10) Pipeline final con XGBoost (pos_weight) + RandomizedSearchCV
#    - pos_weight para desbalance
# ------------------------------------------------------------

# ---- Calcular ratio para scale_pos_weight ----
pos = int(y_train.sum())
neg = int(len(y_train) - pos)
ratio = neg / max(1, pos)

# ---- XGBoost ----
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",           # PR-AUC para desbalance
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    enable_categorical=False
)

pipe_xgb = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", xgb)
])

# ----- Espacio de búsqueda -----
param_dist = {
    "model__n_estimators":      [300, 500, 800, 1000, 1200, 1500],
    "model__max_depth":         [4, 5, 6, 7, 8],
    "model__learning_rate":     [0.01, 0.05, 0.1, 0.15, 0.2],
    "model__subsample":         [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree":  [0.7, 0.8, 0.9, 1.0],
    "model__min_child_weight":  [1, 3, 5, 7],
    "model__gamma":             [0, 0.1, 0.3, 0.5, 1.0],
    "model__reg_alpha":         [0, 0.001, 0.01, 0.1, 1.0],
    "model__reg_lambda":        [0.1, 0.5, 1.0, 2.0, 5.0],
    "model__scale_pos_weight":  [1, ratio, ratio/2, ratio*1.5, np.sqrt(ratio), 10, 15, 20]
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=pipe_xgb,
    param_distributions=param_dist,
    n_iter=167,
    scoring="average_precision",  # métrica principal para elegir el mejor (ideal para clases desbalanceadas)
    n_jobs=-1,
    cv=cv,
    refit=True,
    verbose=1,
    random_state=42
)

# ----- Búsqueda en TRAIN -----
search.fit(X_train, y_train)

print("=== RandomizedSearchCV ===")
print("Best AP (cv):", f"{search.best_score_:.4f}")
print("Best Params:")
for k, v in search.best_params_.items():
    print(f"  {k}: {v}")

best_pipe = search.best_estimator_  # pipeline ya reentrenado en todo TRAIN

# ----- Probabilidades en VALID para elegir umbral por G-Mean -----
p_valid = best_pipe.predict_proba(X_valid)[:, 1]
ap_valid  = average_precision_score(y_valid, p_valid)
roc_valid = roc_auc_score(y_valid, p_valid)

# Buscar umbral óptimo por G-Mean
thresholds = np.linspace(0.01, 0.5, 50)
gmeans, recs, spcs = [], [], []
for t in thresholds:
    g, r, s = gmean_at_threshold(y_valid, p_valid, t)
    gmeans.append(g); recs.append(r); spcs.append(s)

best_idx = int(np.argmax(gmeans))
best_thr = float(thresholds[best_idx])
best_gmean_valid = float(gmeans[best_idx])

# ------------------------------------------------------------
# 11) Evaluación
# ------------------------------------------------------------
yhat_valid = (p_valid >= best_thr).astype(int)
f1_valid = f1_score(y_valid, yhat_valid)

print("\n=== VALIDACIÓN (selección de umbral) ===")
print(f"Average Precision (PR-AUC): {ap_valid:0.4f}")
print(f"ROC-AUC:                   {roc_valid:0.4f}")
print(f"Best threshold (G-Mean):   {best_thr:0.3f}")
print(f"G-Mean @best_thr:          {best_gmean_valid:0.4f}")
print(f"F1-score @best_thr:        {f1_valid:0.4f}")

# --- TEST ---
p_test  = best_pipe.predict_proba(X_test)[:, 1]
ap_test  = average_precision_score(y_test, p_test)
roc_test = roc_auc_score(y_test, p_test)
yhat_test = (p_test >= best_thr).astype(int)
gmean_test, rec_test, spc_test = gmean_at_threshold(y_test, p_test, best_thr)
f1_test  = f1_score(y_test, yhat_test)

print("\n=== PRUEBA (umbral fijado desde VALID) ===")
print(f"Average Precision (PR-AUC): {ap_test:0.4f}")
print(f"ROC-AUC:                    {roc_test:0.4f}")
print(f"G-Mean  @thr={best_thr:0.3f}: {gmean_test:0.4f}")
print(f"  - Recall (TPR):           {rec_test:0.4f}")
print(f"  - Specificity (TNR):      {spc_test:0.4f}")
print(f"F1-score:                   {f1_test:0.4f}")

# Matriz de confusión en test
print("\nConfusion Matrix (Test)")
print(confusion_matrix(y_test, yhat_test, labels=[0,1]))


