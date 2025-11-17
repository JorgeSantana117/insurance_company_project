# MLOPS Project

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

MLOPS project applied to Insurance Company Benchmark (COIL 2000) dataset

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
    │
    └── api                     <- FastAPI service exposing the trained model
        ├── __init__.py
        ├── main.py             <- FastAPI application and route definitions
        ├── service.py          <- Wrapper around the MLflow model (loading + predict)
        └── schemas.py          <- Pydantic request/response schemas
```
Servicio del modelo con API:
Modelo y MLflow

El modelo de clasificación CARAVAN se entrena y se registra en MLflow.

El servicio de API usa por defecto el artefacto:

DEFAULT_MODEL_URI = "runs:/3ba2c4ed3a1f45cb998642e19ed4219a/model"


La URI se puede sobrescribir con la variable de entorno:

export MODEL_URI="runs:/<otro_run_id>/model"
# o un modelo del registry:
export MODEL_URI="models:/insurance_caravan_production/1"


Esto permite portabilidad: en cada entorno (local, Docker, nube) se puede apuntar a una versión distinta del modelo sin cambiar el código.

## Serving the model as an API (FastAPI + MLflow)
### Arquitectura del servicio

src/api/schemas.py

InsuranceRequest: define el schema de entrada (features del modelo).

InsuranceResponse: define la salida (predicted_class, probability, model_uri).

src/api/service.py

Carga el modelo desde MODEL_URI usando mlflow.sklearn.load_model.

Define FEATURE_NAMES para asegurar el orden correcto de columnas.

Implementa InsuranceModelService.predict() que:

Recibe un InsuranceRequest.

Lo convierte a pandas.DataFrame con columnas en FEATURE_NAMES.

Llama a model.predict() y opcionalmente model.predict_proba().

src/api/main.py

Crea la instancia de FastAPI.

Define:

GET /health

POST /predict

###Endpoints disponibles

GET /health
Verifica que el servicio y el modelo estén disponibles.

Ejemplo de respuesta:

{
  "status": "ok",
  "model_uri": "runs:/3ba2c4ed3a1f45cb998642e19ed4219a/model"
}


POST /predict
Recibe un JSON con todas las features definidas en InsuranceRequest y devuelve:

{
  "predicted_class": 0,
  "probability": 0.01816,
  "model_uri": "runs:/3ba2c4ed3a1f45cb998642e19ed4219a/model"
}

###Documentación (OpenAPI / Swagger)

FastAPI expone la documentación automáticamente:

Swagger UI: http://localhost:8000/docs

ReDoc: http://localhost:8000/redoc

Ahí se puede ver el schema de entrada y salida de InsuranceRequest y InsuranceResponse y probar los endpoints sin usar curl.

###Cómo ejecutar la API localmente

Crear y activar el entorno (ejemplo):

conda create -n insurance_mlops python=3.10
conda activate insurance_mlops
pip install -r requirements.txt


Configurar la variable de entorno del modelo:

export MODEL_URI="runs:/3ba2c4ed3a1f45cb998642e19ed4219a/model"


Levantar el servidor FastAPI desde la raíz del proyecto:

uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload


Abrir en el navegador:

http://localhost:8000/docs
 → Swagger UI

http://localhost:8000/health
 → health check

###Ejemplos de uso
Probar con curl
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "num_houses": 2,
  "avg_household_size": 4,
  "avg_age_band": 2,
  "customer_main_type": 3,
  "pct_roman_catholic": 2,
  "pct_protestant": 3,
  "pct_other_religion": 1,
  "pct_no_religion": 5,
  "pct_married": 7,
  "pct_living_together": 0,
  "...": 0
}'


Nota: en la práctica se deben enviar todas las columnas definidas en InsuranceRequest.

###Probar desde Python usando una fila del CSV

Ejemplo de script (src/api/test_request_from_csv.py) que toma la primera fila de un archivo procesado y la envía a la API:

import pandas as pd
import requests

from src.api.service import FEATURE_NAMES

CSV_PATH = "data/processed/insurance_company_cleaned_modified.csv"
API_URL = "http://localhost:8000/predict"

def main():
    df = pd.read_csv(CSV_PATH)

    # Tomamos la primera fila como ejemplo
    row = df.iloc[0]

    # Construimos el payload usando el mismo orden que FEATURE_NAMES
    payload = {col: int(row[col]) for col in FEATURE_NAMES}

    print("Payload que se enviará a la API:")
    for k, v in list(payload.items())[:10]:
        print(f"  {k}: {v}")
    print("  ...")

    resp = requests.post(API_URL, json=payload)

    print("\nStatus code:", resp.status_code)
    print("Respuesta JSON:")
    print(resp.json())

if __name__ == "__main__":
    main()


Ejecutar:

python -m src.api.test_request_from_csv
--------

