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
```

--------

PyTest

Se usa Pyytest para garantizar la fiabilidad del código. Las pruebas están divididas en pruebas unitarias son rápidas, para funciones individuales y pruebas de integración para el flujo completo.

Prerrequisitos

python -m pip install -r requirements.txt

python -m dvc pull


Cómo Ejecutar las Pruebas

Ejecutar todas las pruebas (Unitarias + Integración)

Para ejecutar el conjunto completo de pruebas, utiliza el siguiente comando desde la raíz del proyecto. 

python -m pytest


Ejecutar solo las Pruebas Unitarias 

Validan la lógica interna de los módulos src/.

python -m pytest -m "not integration"


Ejecutar solo la Prueba de Integración

Esta prueba valida el flujo completo E2E (carga, limpieza, entrenamiento, evaluación) y requiere que los datos de DVC (dvc pull) hayan sido descargados.

python -m pytest -m "integration"