import argparse, json
from pathlib import Path
import pandas as pd

from prep_utils import COL_CREATED_AT, COL_CONTEXT, save_parquet

def thread_aware_temporal_split(df: pd.DataFrame, ratios=(0.7,0.15,0.15), created_col=COL_CREATED_AT, thread_col = COL_CONTEXT):
    assert abs(sum(ratios) -1.0) < 1e-6

    g = df.groupby(thread_col)[created_col].min().reset_index().sort_values(created_col)
    n = len(g)
    n_tr = int(n * ratios[0])
    n_dev = int (n * ratios[1])
    train_ids = set(g.iloc[:n_tr][thread_col])
    dev_ids = set(g.iloc[n_tr:n_tr+n_dev][thread_col])
    test_ids = set(g.iloc[n_tr+n_dev:][thread_col])

    train = df[df[thread_col].isin(train_ids)].sort_values(created_col).copy()
    dev = df[df[thread_col].isin(dev_ids)].sort_values(created_col).copy()
    test = df[df[thread_col].isin(test_ids)].sort_values(created_col).copy()

    return train, dev, test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-parquet", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-dev", required=True)
    ap.add_argument("--out-test", required=True)
    ap.add_argument("--ratios", nargs=3, type=float ,default=[0.7,0.15,0.15])
    ap.add_argument("--out-meta", default=None)
    args = ap.parse_args()

    df = pd.read_parquet(args.in_parquet)
    train, dev, test = thread_aware_temporal_split(df, ratios=tuple(args.ratios))

    save_parquet(train, args.out_train)
    save_parquet(dev, args.out_dev)
    save_parquet(test, args.out_test)

    if args.out_meta:
        meta = {
            "counts": {"train": len(train), "dev": len(dev), "test": len(test)},
            "ratios": args.ratios
        }
        Path(args.out_meta).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("OK ->", args.out_train, args.out_dev, args.out_test)

if __name__ == "__main__":
    main()