import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import torch
import numpy as np

MODEL_DIR = Path("BERTolto/models/bertolto_v1")

class VulnDetector:
    def __init__(self, model_dir: Path):
        self.model_dir = Path(model_dir)

        # Config + tokenizer + modelo
        self.conf = AutoConfig.from_pretrained(self.model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir, config=self.conf)
        self.model.eval()

        # Detecta índice de "security"
        self.id2label = {int(k): v for k, v in self.conf.id2label.items()}
        self.label2id = {v: k for k, v in self.id2label.items()}
        
        if "security" not in self.label2id:
            raise ValueError(f"'security' no está en id2label: {self.id2label}")
        
        self.SEC_ID = int(self.label2id["security"])

        thr_path = self.model_dir / "threshold.json"
        thr_json = json.loads(thr_path.read_text(encoding="utf-8"))
        self.threshold = float(thr_json["threshold"])

        # parámetros de troceo (coinciden con los de entrenamiento (tokenizador de comentarios))
        self.max_len = 384
        self.stride = 128

        # versión para trazabilidad
        self.version = {
            "model_dir": str(self.model_dir),
            "security_id": self.SEC_ID,
            "threshold": self.threshold
        }

    def _ensure_item_dict(self, item: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        # Acepta texto crudo o dict con claves {"text", "id", "context_id"}
        if isinstance(item, str):
            return {"text": item, "id": None, "context_id": None}
        return {"text": item.get("text", ""), "id": item.get("id"), "context_id": item.get("context_id")}


    # Predicción/evaluación de texto (comentario)
    @torch.no_grad()
    def predict_text(self, text: str):
        if not text:
            return dict(score_security=0.0, is_security=False, version=self.version)

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_tensors="pt"
        )

        # forward por lotes (por si hay muchos chunks)
        logits_list = []
        bs = 32

        for i in range(0, enc["input_ids"].size(0), bs):
            sl = {k: v[i:i+bs] for k, v in enc.items() if isinstance(v, torch.Tensor)}
            out = self.model(**sl).logits  # [B, num_labels]
            logits_list.append(out.cpu().numpy())

        logits = np.vstack(logits_list)

        # softmax y pooling MAX en la dimensión de chunks
        ex = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = ex / ex.sum(axis=1, keepdims=True)
        p_security = probs[:, self.SEC_ID]
        score = float(p_security.max())  # pooling MAX
        is_sec = bool(score >= self.threshold)

        # dentro de predict_text, justo antes del return
        best_idx = int(np.argmax(p_security))
        best_snippet = self.tokenizer.decode(
            enc["input_ids"][best_idx], skip_special_tokens=True
        )

        # bandas de decisión para triage
        if score >= self.threshold:
            band = "high"
        elif score >= (self.threshold - 0.05):
            band = "mid"
        else:
            band = "low"

        # probs de todas las clases
        class_probs = probs[best_idx].tolist()  # del mejor chunk
        id2label_sorted = [self.id2label[i] for i in range(len(self.id2label))]

        return {
          "score_security": score,
          "is_security": is_sec,
          "band": band,
          "best_snippet": best_snippet[:1000],
          "class_probs": dict(zip(id2label_sorted, class_probs)),
          "version": self.version
        }

        #return dict(score_security=score, is_security=is_sec, version=self.version)

    # Predicción/evaluación de un lote de textos (comentarios)
    @torch.no_grad()
    def predict_batch(self, items: List[Union[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        items: listra de strings o dicts (comentarios) {"text": str, "id": Optional[str], "context_id": Optional[str]}
        Devuelve lista alineada de resultados por item, preservando id/context_id si existen
        Nota. implementación sencilla que aprovecha predict_text() por item individual
        """
        outputs: List[Dict[str, Any]] = []

        for item in items:
            obj = self._ensure_item_dict(item)
            res = self.predict_text(obj["text"])
            res["id"] = obj.get("id")
            res["context_id"] = obj.get("context_id")
            outputs.append(res)

        return outputs


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="Comentario crudo (modo individual)")
    parser.add_argument("--batch_file", help="Ruta a JSON con lista de items {'text', 'id', 'context_id'}")
    parser.add_argument("--model_dir", default=str(MODEL_DIR))
    args = parser.parse_args()

    det = VulnDetector(Path(args.model_dir))
    res = det.predict_text(args.text)

    print(json.dumps(res, ensure_ascii=False, indent=2))

    # El funcionamiento interno es idéntico, se recibe un batch de comentarios y cada elemento de la lista se evalua individualmente
    # El objetivo de este modo es reducir las llamadas a la API
    if args.batch_file:
        items = json.loads(Path(args.batch_file).read_text(encoding="utf-8"))
        res = det.predict_batch(items)
    else:
        assert args.text is not None, "Debes pasar --text o --batch_file"
        res = det.predict_text(args.text)

    print(json.dumps(res, ensure_ascii=False, indent=2))