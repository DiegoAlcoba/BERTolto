import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
import os
from datasets import Dataset, DatasetDict
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, DataCollatorWithPadding  # DataCollator import se mantiene por compat

from prep_utils import (
    COL_ID, COL_TEXT, COL_LABEL, COL_SOURCE, COL_CREATED_AT, COL_CONTEXT
)

# --- NUEVO: Defaults seguros para cache HF si no están definidos por entorno ---
# Evita usar ~/.cache/huggingface (que puede quedar con permisos de root cuando usas contenedor)
_HF_CACHE_DEFAULT = Path(__file__).resolve().parents[2] / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(_HF_CACHE_DEFAULT))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_HF_CACHE_DEFAULT))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_HF_CACHE_DEFAULT / "hub"))
Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["TRANSFORMERS_CACHE"]).mkdir(parents=True, exist_ok=True)

_TOKENIZER_CACHE = {}

# Función de tokenización (con/sin ventanas)
def tokenize_batch(batch, base_model, max_len, sliding_window, slide_stride, max_window_per_doc, use_domain_prefix):
    # Lazy: cada proceso creará su tokenizer la primera vez
    key = (base_model, use_domain_prefix, max_len)
    tok = _TOKENIZER_CACHE.get(key)

    if tok is None:
        tok = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        tok.model_max_length = max_len
        if use_domain_prefix:
            tok.add_special_tokens({"additional_special_tokens": ["<GITHUB>", "<REDDIT>"]})
        _TOKENIZER_CACHE[key] = tok

    # Tokenización con/sin overflow
    enc = tok(
        batch[COL_TEXT],
        truncation=True,
        padding=False,
        max_length=max_len,
        return_overflowing_tokens=sliding_window,
        stride=(slide_stride if sliding_window else 0),
    )

    mapping = enc.pop("overflow_to_sample_mapping", None)

    if mapping is None:
        # Sin overflow: 1 salida por ejemplo de entrada
        src_idx = list(range(len(batch[COL_TEXT])))
    else:
        src_idx = list(mapping)

    if sliding_window and (max_window_per_doc is not None) and (mapping is not None):
        keep = []
        counts = {}
        for i, src in enumerate(src_idx):
            c = counts.get(src, 0)
            if c < max_window_per_doc:
                keep.append(i); counts[src] = c + 1
        if len(keep) != len(src_idx):
            for k in list(enc.keys()):
                enc[k] = [enc[k][i] for i in keep]
            src_idx = [src_idx[i] for i in keep]

    # Replicar columnas meta para que tengan la MISMA longitud que input_ids/attention_mask
    out = dict(enc)
    out[COL_ID] = [batch[COL_ID][i] for i in src_idx]
    out[COL_LABEL] = [batch[COL_LABEL][i] for i in src_idx]
    out[COL_SOURCE] = [batch[COL_SOURCE][i] for i in src_idx]
    out[COL_CREATED_AT] = [batch[COL_CREATED_AT][i] for i in src_idx]
    out[COL_CONTEXT] = [batch[COL_CONTEXT][i] for i in src_idx]

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parquet", required=True)
    ap.add_argument("--dev-parquet", required=True)
    ap.add_argument("--test-parquet", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-model", default="distilroberta-base")
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--use-domain-prefix", action="store_true")
    ap.add_argument("--sliding-window", action="store_true")
    ap.add_argument("--slide-stride", type=int, default=128)

    # --- NUEVO: acepta ambas variantes y unifica en 'max_windows_per_doc'
    ap.add_argument(
        "--max-windows-per-doc", "--max-window-per-doc",
        dest="max_windows_per_doc",
        type=int, default=None,
        help="Máximo de ventanas por documento cuando hay sliding-window."
    )

    ap.add_argument("--filter-max-input-tokens", type=int, default=None)
    ap.add_argument("--num-proc", type=int, default=max(1, (os.cpu_count() or 4)//2))
    ap.add_argument("--batch-chunk-size", type=int, default=512)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    keep_cols = [COL_ID, COL_TEXT, COL_LABEL, COL_SOURCE, COL_CREATED_AT, COL_CONTEXT]
    train = pd.read_parquet(args.train_parquet)[keep_cols].copy()
    dev = pd.read_parquet(args.dev_parquet)[keep_cols].copy()
    test = pd.read_parquet(args.test_parquet)[keep_cols].copy()

    # Prefijo de dominio (opcional)
    if args.use_domain_prefix:
        def pref(src: str) -> str:
            s = str(src).lower()
            return "<GITHUB> " if "github" in s else ("<REDDIT> " if "reddit" in s else "")

        for df in (train, dev, test):
            df[COL_TEXT] = df.apply(lambda r: pref(r[COL_SOURCE]) + r[COL_TEXT], axis=1)

    # Tokenizador de Transformers
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tok.model_max_length = args.max_len

    if args.use_domain_prefix:
        tok.add_special_tokens({"additional_special_tokens": ["<GITHUB>", "<REDDIT>"]})

    # Calcular longitudes en tokens en batch (sin truncar)
    def measure_lengths(texts, tokenizer, bs):
        lens=[]
        for i in range(0, len(texts), bs):
            enc = tokenizer(
                list(texts[i:i+bs]), truncation=False,
                add_special_tokens=True, padding=False
            )
            lens.extend([len(x) for x in enc["input_ids"]])
        return np.array(lens, dtype=np.int32)

    # Filtro (opcional) antes de la división por ventanas
    if args.filter_max_input_tokens is not None:
        TH = int(args.filter_max_input_tokens)
        for name, df in (("train", train), ("validation", dev), ("test", test)):
            if len(df) == 0:
                continue
            lens = measure_lengths(df[COL_TEXT].astype(str).tolist(), tok, args.batch_chunk_size)
            keep_mask = lens <= TH
            kept = keep_mask.sum()
            if kept < len(df):
                print(f"[filtro] {name} -> {len(df) - kept} descartados por > {TH} tokens")
            if name == "train":
                train = df.iloc[np.where(keep_mask)[0]].copy()
            elif name == "validation":
                dev = df.iloc[np.where(keep_mask)[0]].copy()
            else:
                test = df.iloc[np.where(keep_mask)[0]].copy()

    # HF Datasets
    dsd = DatasetDict({
        "train": Dataset.from_pandas(train, preserve_index=False),
        "validation": Dataset.from_pandas(dev, preserve_index=False),
        "test": Dataset.from_pandas(test, preserve_index=False)
    })

    try:
        ds_tok = dsd.map(
            tokenize_batch,
            batched=True,
            remove_columns=[COL_ID, COL_TEXT, COL_LABEL, COL_SOURCE, COL_CREATED_AT, COL_CONTEXT],
            batch_size=args.batch_chunk_size,
            num_proc=args.num_proc,
            fn_kwargs=dict(
                base_model=args.base_model,
                max_len=args.max_len,
                sliding_window=args.sliding_window,
                slide_stride=args.slide_stride,
                # --- mapping al nombre que usa tokenize_batch (sin cambiar su firma)
                max_window_per_doc=args.max_windows_per_doc,
                use_domain_prefix=args.use_domain_prefix,
            )
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise

    # Guardar dataset y tokenizador
    hf_ds_dir = out_dir / "dataset"
    tok_dir = out_dir / "tokenizer"
    ds_tok.save_to_disk(str(hf_ds_dir))
    tok.save_pretrained(str(tok_dir))

    # Class weights (0/1) por si se entrena luego con 'balanced'
    y = np.array(ds_tok["train"][COL_LABEL])
    classes = np.unique(y)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    cw_map = {int(c): float(w) for c, w in zip(classes, cw)}

    meta = {
        "base_model": args.base_model,
        "max_len": args.max_len,
        "use_domain_prefix": bool(args.use_domain_prefix),
        "sliding_window": bool(args.sliding_window),
        "slide_stride": args.slide_stride,
        "counts": {k: len(ds_tok[k]) for k in ds_tok.keys()},
        "class_weights": cw_map
    }

    (out_dir / "preprocess_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("OK ->", hf_ds_dir, tok_dir)

if __name__ == "__main__":
    main()
