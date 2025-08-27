import argparse, json
from pathlib import Path
import pandas as pd

from prep_utils import (COL_ID, COL_TEXT, COL_LABEL, COL_SOURCE, COL_CONTEXT, COL_CREATED_AT,
                        read_any, ensure_columns, normalize_basic, load_sqlite, save_parquet)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", help="Rutas CSV(JSONL (0..N.")
    ap.add_argument("--sqlite-db", type=str, help = "Ruta a SQLite (opcional)")
    ap.add_argument("--table", type=str, help = "Tabla SQLite (si no se especifica --query)")
    ap.add_argument("--query", type=str, help = "SELECT custom para SQLite")
    ap.add_argument("--limit", type=int, help = "Limitar filas (debug)")
    ap.add_argument("--out-parquet", required=True, help = "Salida merged.parquet")
    ap.add_argument("--out-meta", required=False, help = "Salida metadata.json")
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
        # debería traer core, pero por asegurar
        sdf = ensure_columns(sdf)
        frames.append(sdf)

    if not frames:
        raise SystemExit("No hay datos de entrada. Usa --inputs o --sqlite-db.")

    big = pd.concat(frames, ignore_index=True)

    # Dedupe por id
    before = len(big)
    big = big.drop_duplicates(subset=[COL_ID], keep="first")

    if len(big) != before:
        print(f"[dedupe] {before - len(big)} duplicados por id eliminados")

    # Normalización ligera
    big = normalize_basic(big)

    # Guardado
    save_parquet(big, args.out_parquet)

    if args.out_meta:
        meta = {
            "rows": len(big),
            "columns": [COL_ID, COL_TEXT, COL_LABEL, COL_SOURCE, COL_CREATED_AT, COL_CONTEXT]
        }
        Path(args.out_meta).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_meta).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("OK ->", args.out_parquet)

if __name__ == "__main__":
    main()


















