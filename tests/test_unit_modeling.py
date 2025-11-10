import pytest
import pandas as pd
import numpy as np
from src.modeling.train import split_data, _specificity_score

# --- Fixture para datos de prueba de splitting ---

@pytest.fixture
def data_for_splitting():
    #Crea un DataFrame simple para probar la división de datos
    X = pd.DataFrame({'feature': np.arange(100)})
    # Crear un target desbalanceado (90 ceros, 10 unos)
    y = pd.Series([0]*90 + [1]*10, name="target")
    return X, y

# Pruebas para split_data

def test_split_data_shapes(data_for_splitting):
    """
    Prueba que la función split_data retorna DataFrames con los shapes correctos.
    (60% train, 20% valid, 20% test)
    """
    X, y = data_for_splitting
    total_rows = len(X) # 100
    
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(
        X, y, random_state=42
    )
    
    # Verificar los shapes
    assert len(X_train) == int(total_rows * 0.60) # 60
    assert len(y_train) == int(total_rows * 0.60) # 60
    
    assert len(X_valid) == int(total_rows * 0.20) # 20
    assert len(y_valid) == int(total_rows * 0.20) # 20
    
    assert len(X_test) == int(total_rows * 0.20) # 20
    assert len(y_test) == int(total_rows * 0.20) # 20
    
    print("\nSplit de datos: Shapes (60/20/20) correctos.")

def test_split_data_stratification(data_for_splitting):
    """
    Prueba que la división (split) fue estratificada.
    El ratio de 'unos' (10%) debe mantenerse en todos los splits.
    """
    X, y = data_for_splitting
    
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(
        X, y, random_state=42
    )
    
    # Ratio original: 10% (10/100)
    original_ratio = y.mean()
    
    # Ratios en los splits (deben ser aprox. 0.10)
    # y_train (60 filas) -> 6 unos
    # y_valid (20 filas) -> 2 unos
    # y_test (20 filas) -> 2 unos
    assert np.isclose(y_train.mean(), original_ratio)
    assert np.isclose(y_valid.mean(), original_ratio)
    assert np.isclose(y_test.mean(), original_ratio)
    
    print("\nSplit de datos: Estratificación correcta.")

# Pruebas para métricas personalizadas (_specificity_score)

def test_specificity_score():
    """Prueba la métrica de especificidad (True Negative Rate)."""
    
    # y_true: [TN, FP, FN, TP]
    # Caso 1: Perfecto
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    # TN=2, FP=0. Especificidad = 2 / (2 + 0) = 1.0
    assert np.isclose(_specificity_score(y_true, y_pred), 1.0)
    
    # Caso 2: Pobre
    y_true = [0, 0, 1, 1]
    y_pred = [1, 1, 0, 0] # Todo mal
    # TN=0, FP=2. Especificidad = 0 / (0 + 2) = 0.0
    assert np.isclose(_specificity_score(y_true, y_pred), 0.0)
    
    # Caso 3: Mixto
    y_true = [0, 0, 0, 1, 1, 1] # 3 Negativos, 3 Positivos
    y_pred = [0, 1, 0, 1, 1, 0] # 2 TN, 1 FP, 1 FN, 2 TP
    # TN=2, FP=1. Especificidad = 2 / (2 + 1) = 0.666...
    assert np.isclose(_specificity_score(y_true, y_pred), 2/3)
    
    print("\nMétrica _specificity_score: Cálculos correctos.")