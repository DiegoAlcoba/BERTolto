import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, DataCollatorWithPadding

from prep_utils import (
    COL_ID, COL_TEXT, COL_LABEL, COL_SOURCE, COL_CREATED_AT, COL_CONTEXT
)

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
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    keep = {COL_ID, COL_TEXT, COL_LABEL, COL_SOURCE, COL_CREATED_AT, COL_CONTEXT}
    train = pd.read_parquet(args.train_parquet)[keep].copy()
    dev = pd.read_parquet(args.dev_parquet)[keep].copy()
    test = pd.read_parquet(args.test_parquet)[keep].copy()

    if args.use_domain_prefix:
        def pref(src: str) -> str:
            s = str(src).lower()

            return "<GITHUB> " if "github" in s else ("<REDDIT> " if "reddit" in s else "")

        for df in (train, dev, test):
            df[COL_TEXT] = df.apply(lambda r: pref(r[COL_SOURCE]) + r[COL_TEXT], axis=1)

    dsd = DatasetDict({
        "train": Dataset.from_pandas(train, preserve_index=False),
        "validation": Dataset.from_pandas(dev, preserve_index=False),
        "test": Dataset.from_pandas(test, preserve_index=False)
    })

    # Tokenizador de transformers
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    if args.use_domain_prefix:
        tok.add_special_tokens({"additional_special_tokens": ["<GITHUB>", "<REDDIT>"]})

    def tokenize_batch(batch):
        return tok(batch[COL_TEXT], truncation=True, max_length=args.max_len)

    def tokenize_sliding(batch):
        return tok(batch[COL_TEXT], truncation=False, add_special_tokens=True, return_overflowing_tokens=True,
                   stride=args.slide_stride, max_length=args.max_len)

    if args.sliding_window:
        ds_tok = dsd.map(tokenize_sliding, batched=True, remove_columns=[COL_TEXT], batch_size=512)
    else:
        ds_tok = dsd.map(tokenize_batch, batched=True, remove_columns=[COL_TEXT], batch_size=1024)

    # Guardados
    hf_ds_dir = out_dir / "dataset"
    tok_dir = out_dir / "tokenizer"
    ds_tok.save_to_disk(str(hf_ds_dir))
    tok.save_pretrained(str(tok_dir))

    #Class weights (0/1) por si se va a utilizar en el entrenamiento
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


















