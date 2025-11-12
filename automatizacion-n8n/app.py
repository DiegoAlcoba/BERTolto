# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from infer.py import VulnDetector, Path

app = FastAPI(title="vuln-detector")
det = VulnDetector(Path("BERTolto/models/vuln_detector_v1"))

class Item(BaseModel):
    id: str | None = None
    context_id: str | None = None
    text: str

@app.post("/predict")
def predict(item: Item):
    out = det.predict_text(item.text)
    return {
        "id": item.id,
        "context_id": item.context_id,
        **out
    }