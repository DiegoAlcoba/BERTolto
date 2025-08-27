import argparse
import pandas as pd
from prep_utils import COL_LABEL, COL_SOURCE, COL_CREATED_AT

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-parquet", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.in_parquet)
    print("rows:", len(df))

    if COL_LABEL in df.columns:
        print("labels:", df[COL_LABEL].value_counts(dropna=False).to_dict())
    if COL_SOURCE in df.columns:
        print("fuentes:", df[COL_SOURCE].value_counts(dropna=False).to_dict())
    if COL_CREATED_AT in df.columns:
        print("rango de fechas:", df[COL_CREATED_AT].min(), "->", df[COL_CREATED_AT].max())

if __name__ == "__main__":
    main()