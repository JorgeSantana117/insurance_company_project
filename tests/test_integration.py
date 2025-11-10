"""
Prueba de Integración: Valida el flujo E2E (End-to-End).
Carga -> Preprocesa -> Entrena -> Evalúa
"""

import pytest
import pandas as pd
import numpy as np

# Importar las funciones refactorizadas
from src import config
from src import dataset
from src import features
from src.modeling import train
import mlflow

# --- Configuración de la Prueba de Integración ---

# Usar @pytest.mark.skip para evitar que esta prueba se ejecute
# cada vez, ya que puede ser lenta y requiere los datos de DVC.
# Para ejecutarla: pytest -m integration
@pytest.mark.integration
def test_full_pipeline_run():

    print("\n--- Iniciando Prueba de Integración E2E ---")
    
    # Definir un nombre de experimento de prueba para MLflow
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME + "_Test")
    run_name = "IntegrationTest_LR"
    
    # Carga de Configuración
    assert config.RAW_DATA_PATH.exists(), "El archivo de datos crudos (config) no se encuentra."
    assert config.PROCESSED_DATA_PATH.exists(), "El archivo de datos procesados (config) no se encuentra."
    
    # Limpieza de Datos (dataset.py)
    print("Ejecutando: dataset.clean_raw_data()")
    try:
        dataset.clean_raw_data(
            in_path=config.RAW_DATA_PATH,
            out_path=config.PROCESSED_DATA_PATH,
            columns=config.COLS,
            expected_ranges=config.EXPECTED_RANGES,
            rare_threshold=config.RARE_THRESHOLD
        )
    except Exception as e:
        pytest.fail(f"Fallo en dataset.clean_raw_data(): {e}")

    # Carga de Datos Procesados (dataset.py)
    print("Ejecutando: dataset.load_processed_data()")
    try:
        df_cleaned = dataset.load_processed_data(config.PROCESSED_DATA_PATH)
        df_cleaned.columns = config.LONG_NAMES
    except Exception as e:
        pytest.fail(f"Fallo en dataset.load_processed_data(): {e}")
        
    assert isinstance(df_cleaned, pd.DataFrame)
    assert not df_cleaned.empty

    # 4. Pruning (features.py)
    print("Ejecutando: features.prune_by_correlation()")
    df_pruned, _ = features.prune_by_correlation(
        df_cleaned,
        target_col=config.TARGET_COL,
        thr=config.PRUNING_THRESHOLD,
        random_state=config.RANDOM_STATE
    )
    assert df_pruned.shape[1] < df_cleaned.shape[1] # Verificar que se eliminaron columnas

    # 5. Definir X/y
    y = df_pruned[config.TARGET_COL].astype(int)
    X = df_pruned.drop(columns=[config.TARGET_COL])

    # 6. Split (train.py)
    print("Ejecutando: train.split_data()")
    X_train, X_valid, X_test, y_train, y_valid, y_test = train.split_data(
        X, y, random_state=config.RANDOM_STATE
    )
    assert len(X_train) + len(X_valid) + len(X_test) == len(X)

    # 7. Preprocesador (features.py)
    print("Ejecutando: features.create_preprocessor()")
    preprocessor = features.create_preprocessor(
        available_cols=X.columns,
        static_nominal_cols=config.NOMINAL_COLS,
        static_ordinal_cols=config.ORDINAL_COLS,
        static_numeric_cols=config.NUMERIC_DISCRETE_COLS
    )
    assert preprocessor is not None

    # 8. Entrenamiento y Evaluación (train.py)
    print(f"Ejecutando: train.train_logistic_regression() con MLflow Run '{run_name}'")
    try:
        lr_results = train.train_logistic_regression(
            X_train, y_train, 
            X_valid, y_valid, 
            X_test, y_test, 
            preprocessor,
            random_state=config.RANDOM_STATE,
            mlflow_run_name=run_name # <--- Integración con MLflow
        )
    except Exception as e:
        pytest.fail(f"Fallo en train.train_logistic_regression(): {e}")

    assert isinstance(lr_results, dict)
    assert 'model' in lr_results
    assert 'test_pr_auc' in lr_results
    assert 'mlflow_run_id' in lr_results
    assert lr_results['test_pr_auc'] > 0 # Debe ser mejor que adivinar al azar
    
    print(f"\nPrueba de Integración E2E completada con éxito.")
    print(f"Resultados del Baseline (Test PR AUC): {lr_results['test_pr_auc']:.4f}")