import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from infer import VulnDetector

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/bertolto_v1")

app = FastAPI(title="BERTolto vulnerability detector", version="1.0")
detector = None

class PredictItem(BaseModel):
    text: str = Field(..., description="Texto del comentario")
    id: Optional[str] = Field(None, description="Identificador del comentario")
    context_id: Optional[str] = Field(None, description="Identificador del contexto (issue/PR, etc.)")

class PredictBatchRequest(BaseModel):
    items: List[PredictItem]

class PredictResponseItem(BaseModel):
    id: Optional[str]
    context_id: Optional[str]
    score_security: float
    is_security: bool
    band: str
    best_snippet: str
    class_probs: dict
    version: dict

class PredictBatchResponse(BaseModel):
    results: List[PredictResponseItem]

@app.on_event("startup")
def load_model():
    global detector
    try:
        detector = VulnDetector(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar el modelo en {MODEL_PATH}: {e}")
    
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict", response_model=PredictResponseItem)
def predict(item: PredictItem):
    if detector is None:
        raise HTTPException(status_code=500, detail="Modelo no inicializado")
    
    # Reutiliza batch con 1 elemento (misma salida que /predict_batch)
    out = detector.predict_batch([item.model_dump()])[0]

    return out

@app.post("/predict_batch", response_model=PredictBatchResponse)
def predict_batch(req: PredictBatchRequest):
    if detector is None:
        raise HTTPException(status_code=500, detail="Modelo no inicializado")
    
    outs = detector.predict_batch([it.model_dump() for it in req.items])

    return {"results": outs}