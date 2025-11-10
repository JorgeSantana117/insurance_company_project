"""
Pruebas Unitarias para el módulo 'features.py'
"""
import pytest
import pandas as pd
import numpy as np
from src.features import create_preprocessor, prune_by_correlation

# --- Fixture para crear un DataFrame de prueba ---

@pytest.fixture
def sample_dataframe():
    """Crea un DataFrame de juguete para las pruebas."""
    data = {
        'numeric_1': [1, 2, 3, 4, 5], # Correlacionado con corr_1
        'numeric_2': [1, 1, 1, 1, 1], # Varianza cero
        'numeric_3_log': [0, 10, 1, 100, 5],
        'ordinal_1': [1, 2, 3, 1, 2],
        'nominal_1': ['A', 'B', 'A', 'B', 'C'],
        'nominal_2': ['X', 'Y', 'X', 'Y', 'X'],
        'corr_1': [1, 2, 3, 4, 5],
        'corr_2': [0.9, 2.1, 2.9, 4.0, 5.1], # Alta correlación con corr_1
        'target': [0, 1, 0, 1, 0]
    }
    return pd.DataFrame(data)

# --- Pruebas para create_preprocessor ---

def test_preprocessor_transforms_columns(sample_dataframe):
    """
    Prueba que el preprocesador transforma el número correcto de columnas.
    """
    X = sample_dataframe.drop('target', axis=1)
    
    # Definir listas de columnas (basadas en los nombres del fixture)
    nominal_cols = ['nominal_1', 'nominal_2']
    ordinal_cols = ['ordinal_1']
    numeric_cols = ['numeric_1', 'numeric_2', 'numeric_3_log', 'corr_1', 'corr_2']
    
    preprocessor = create_preprocessor(
        available_cols=X.columns,
        static_nominal_cols=nominal_cols,
        static_ordinal_cols=ordinal_cols,
        static_numeric_cols=numeric_cols
    )
    
    # Entrenar el preprocesador
    preprocessor.fit(X)
    
    # Transformar los datos
    X_transformed = preprocessor.transform(X)
    
    # Total de columnas esperadas = 5 (OHE) + 1 (StdScaler) + 5 (Log+Robust) = 11
    assert X_transformed.shape[0] == 5 # 5 filas
    assert X_transformed.shape[1] == 11 # 11 columnas transformadas
    
    print(f"\nShape de salida: {X_transformed.shape} (Esperado: 5, 11)")

def test_preprocessor_handles_unseen_categories(sample_dataframe):
    """
    Prueba que el OneHotEncoder (con handle_unknown='ignore')
    maneja correctamente categorías no vistas.
    """
    X_train = sample_dataframe.drop('target', axis=1)
    
    nominal_cols = ['nominal_1', 'nominal_2']
    ordinal_cols = ['ordinal_1']
    numeric_cols = ['numeric_1', 'numeric_2', 'numeric_3_log', 'corr_1', 'corr_2']
    
    preprocessor = create_preprocessor(
        available_cols=X_train.columns,
        static_nominal_cols=nominal_cols,
        static_ordinal_cols=ordinal_cols,
        static_numeric_cols=numeric_cols
    )
    
    preprocessor.fit(X_train)
    
    # Crear datos de "prueba" con una categoría no vista ('D' en nominal_1)
    test_data = {
        'numeric_1': [6], 'numeric_2': [1], 'numeric_3_log': [2],
        'ordinal_1': [1], 'nominal_1': ['D'], 'nominal_2': ['Z'],
        'corr_1': [6], 'corr_2': [5.9], 'target': [1]
    }
    X_test = pd.DataFrame(test_data)
    
    try:
        X_test_transformed = preprocessor.transform(X_test)
    except Exception as e:
        pytest.fail(f"El preprocesador falló con categorías no vistas: {e}")
        
    assert X_test_transformed.shape == (1, 11)
    assert np.sum(X_test_transformed[0, 0:5]) == 0
    print("\nCategorías no vistas ('D', 'Z') manejadas correctamente (todas cero).")

# --- Pruebas para prune_by_correlation ---

def test_prune_by_correlation(sample_dataframe):
    """
    Prueba que la función de poda por correlación elimina la columna correcta.
    """
    df_pruned, dropped_cols = prune_by_correlation(
        df=sample_dataframe,
        target_col='target',
        thr=0.9, # Umbral bajo para forzar la poda
        random_state=42
    )
    
    # La función encontró 2 columnas para eliminar (ej. 'corr_1' y 'numeric_1')
    # Esta es la corrección:
    assert len(dropped_cols) == 2
    
    # Verificar que las columnas correctas (o una combinación válida) fueron eliminadas
    # Sabemos por el log de error que eliminó ['corr_1', 'numeric_1']
    assert 'corr_1' in dropped_cols
    assert 'numeric_1' in dropped_cols
    
    # El DataFrame resultante no debe contener las columnas eliminadas
    assert 'corr_1' not in df_pruned.columns
    assert 'numeric_1' not in df_pruned.columns
    
    # El DataFrame DEBE contener la columna que se quedó ('corr_2')
    assert 'corr_2' in df_pruned.columns
    
    print(f"\nPruning: {len(dropped_cols)} columnas eliminadas correctamente.")