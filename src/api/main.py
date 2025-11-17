# src/api/main.py
from fastapi import FastAPI, HTTPException

from .schemas import InsuranceRequest, InsuranceResponse
from .service import service

app = FastAPI(
    title="Insurance Company CARAVAN API",
    description=(
        "Servicio FastAPI para exponer el modelo de clasificación CARAVAN "
        "entrenado y registrado con MLflow."
    ),
    version="1.0.0",
)


@app.get("/health", tags=["health"])
def health_check():
    """
    Verifica que el servicio y el modelo estén disponibles.
    """
    return {
        "status": "ok",
        "model_uri": service.model_uri,
    }


@app.post("/predict", response_model=InsuranceResponse, tags=["prediction"])
def predict(request: InsuranceRequest):
    """
    Endpoint principal de predicción.
    """
    try:
        result = service.predict(request)
        return InsuranceResponse(**result)
    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Falta la columna requerida en el request: {e}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
