import argparse, json
from pathlib import Path
import pandas as pd
import re

from prep_utils import (
    COL_ID, COL_TEXT, COL_LABEL, COL_SOURCE, COL_CONTEXT, COL_CREATED_AT,
    read_any, ensure_columns, normalize_basic, load_sqlite, save_parquet
)

# Heurísticas para etiquetado débil
BUG_HINTS = {"kind/bug", "sig/bug", "triage/accepted", "area/bug"}
SEC_HINTS = {"sig/security", "area/security", "kind/security"}

def _build_context(row) -> str:
    title = (row.get("container_title") or "").strip()
    labels = (row.get("container_labels") or "").strip()
    if labels:
        parts = [s for s in labels.split(";") if s]
        lab = "".join(f"[{s}]" for s in parts)
        return f"{title} {lab}".strip()
    return title

def _weak_label(row) -> str:
    labs = set(s.strip().lower() for s in (row.get("container_labels") or "").split(";") if s)
    title = (row.get("container_title") or "").lower()
    text  = (row.get("text") or "").lower()
    if (labs & SEC_HINTS) or any(k in title or k in text for k in ["cve-","vulnerability","exploit","rce","xss","csrf","ssrf"]):
        return "security"
    if (labs & BUG_HINTS) or ("bug" in title):
        return "bug"
    return "other"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", help="Rutas CSV/JSONL (0..N).")
    ap.add_argument("--sqlite-db", type=str, help="Ruta a SQLite (opcional)")
    ap.add_argument("--table", type=str, help="Tabla SQLite (si no se especifica --query)")
    ap.add_argument("--query", type=str, help="SELECT custom para SQLite")
    ap.add_argument("--limit", type=int, help="Limitar filas (debug)")
    ap.add_argument("--out-parquet", required=True, help="Salida merged.parquet")
    ap.add_argument("--out-meta", required=False, help="Salida metadata.json")
    ap.add_argument("--apply-noise-filter", action="store_true", help="Filtra bots/plantillas/comentarios cortos.")
    ap.add_argument("--build-core-schema", action="store_true", help="Construye columnas core (context,label,source,id,created_at).")
    ap.add_argument("--drop-empty-text", action="store_true", help="Descarta filas con texto vacío o NaN.")
    ap.add_argument("--source-default", type=str, default=None, help='Valor por defecto para COL_SOURCE (p.ej. "<GITHUB>").')
    args = ap.parse_args()

    frames = []

    # Inputs de archivo
    for p in (args.inputs or []):
        df = read_any(p)
        df = ensure_columns(df)
        frames.append(df)

    # SQLite (opcional)
    if args.sqlite_db:
        sdf = load_sqlite(args.sqlite_db, query=args.query, table=args.table, limit=args.limit)
        sdf = ensure_columns(sdf)
        frames.append(sdf)

    if not frames:
        raise SystemExit("No hay datos de entrada. Usa --inputs o --sqlite-db.")

    big = pd.concat(frames, ignore_index=True)

    # Mapeo de columnas cuando vienen del CSV de GitHub
    if "comment_id" in big.columns and COL_ID not in big.columns:
        big[COL_ID] = big["comment_id"]
    if "text" in big.columns and COL_TEXT not in big.columns:
        big[COL_TEXT] = big["text"]
    if "comment_created_at" in big.columns and COL_CREATED_AT not in big.columns:
        big[COL_CREATED_AT] = big["comment_created_at"]
    if COL_SOURCE not in big.columns and args.source_default:
        big[COL_SOURCE] = args.source_default

    # Dedupe por id
    before = len(big)
    big = big.drop_duplicates(subset=[COL_ID], keep="first")
    if len(big) != before:
        print(f"[dedupe] {before - len(big)} duplicados por id eliminados")

    # Normalización ligera de texto
    big = normalize_basic(big)

    # Eliminar textos vacíos (opcional)
    if args.drop_empty_text and COL_TEXT in big.columns:
        b0 = len(big)
        big[COL_TEXT] = big[COL_TEXT].astype(str)
        big = big[big[COL_TEXT].str.strip().astype(bool)]
        print(f"[drop-empty-text] {b0 - len(big)} filas sin texto eliminadas")

    # Construcción del esquema core
    if args.build_core_schema:
        if COL_CONTEXT not in big.columns:
            if {"container_title","container_labels"}.issubset(big.columns):
                big[COL_CONTEXT] = big.apply(_build_context, axis=1)
            else:
                big[COL_CONTEXT] = ""

        if COL_LABEL not in big.columns:
            if {"container_title","container_labels", COL_TEXT}.issubset(big.columns):
                big[COL_LABEL] = big.apply(_weak_label, axis=1)
            else:
                big[COL_LABEL] = "other"

        if COL_SOURCE not in big.columns:
            big[COL_SOURCE] = args.source_default or "<GITHUB>"

        if COL_ID not in big.columns and "id" in big.columns:
            big[COL_ID] = big["id"]

        if COL_CREATED_AT in big.columns:
            big[COL_CREATED_AT] = pd.to_datetime(big[COL_CREATED_AT], errors="coerce", utc=True)
        elif "container_created_at" in big.columns:
            big[COL_CREATED_AT] = pd.to_datetime(big["container_created_at"], errors="coerce", utc=True)

    # Asegurar tipo datetime UTC si existe la columna
    if COL_CREATED_AT in big.columns and str(big[COL_CREATED_AT].dtype) != "datetime64[ns, UTC]":
        big[COL_CREATED_AT] = pd.to_datetime(big[COL_CREATED_AT], errors="coerce", utc=True)

    # Guardado
    save_parquet(big, args.out_parquet)

    if args.out_meta:
        meta = {"rows": len(big)}
        meta["columns_present"] = [c for c in [COL_ID, COL_TEXT, COL_LABEL, COL_SOURCE, COL_CREATED_AT, COL_CONTEXT] if c in big.columns]
        if COL_CREATED_AT in big.columns:
            try:
                meta["created_at_range"] = [str(big[COL_CREATED_AT].min()), str(big[COL_CREATED_AT].max())]
            except Exception:
                pass
        if COL_LABEL in big.columns:
            meta["label_dist"] = big[COL_LABEL].value_counts(dropna=False).to_dict()
        if COL_SOURCE in big.columns:
            meta["source_dist"] = big[COL_SOURCE].value_counts(dropna=False).to_dict()
        Path(args.out_meta).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_meta).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("OK ->", args.out_parquet)

if __name__ == "__main__":
    main()
