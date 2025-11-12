import argparse
import pandas as pd
import numpy as np
from pandas.api.types import is_datetime64_any_dtype

# nombres canónicos (ajústalos si en prep_utils tienes constantes diferentes)
COL_LABEL = "label"
COL_SOURCE = "source"
COL_CREATED_AT = "created_at"
COL_REPO = "repo"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-parquet", required=True)
    ap.add_argument("--topn", type=int, default=10, help="Top-N repos por volumen")
    args = ap.parse_args()

    df = pd.read_parquet(args.in_parquet)

    print(f"rows: {len(df)}")

    if COL_LABEL in df.columns:
        print("labels:", df[COL_LABEL].value_counts(dropna=False).to_dict())

    if COL_SOURCE in df.columns:
        print("fuentes:", df[COL_SOURCE].value_counts(dropna=False).to_dict())

    if COL_CREATED_AT in df.columns:
        # Asegura datetime (acepta tz-aware)
        if not is_datetime64_any_dtype(df[COL_CREATED_AT]):
            df[COL_CREATED_AT] = pd.to_datetime(df[COL_CREATED_AT], errors="coerce", utc=True)

        # por si hay NaT
        rng_min = df[COL_CREATED_AT].min()
        rng_max = df[COL_CREATED_AT].max()

        # imprime en ISO (maneja NaT)
        min_str = rng_min.isoformat() if pd.notna(rng_min) else "NaT"
        max_str = rng_max.isoformat() if pd.notna(rng_max) else "NaT"
        print("rango de fechas:", min_str, "->", max_str)

    # Top repos
    if (COL_REPO in df.columns) and (args.topn and args.topn > 0):
        vc = df[COL_REPO].value_counts().head(args.topn)
        if not vc.empty:
            print(f"top {args.topn} repos:")
            for repo, cnt in vc.items():
                print(f"  - {repo}: {cnt}")

if __name__ == "__main__":
    main()
