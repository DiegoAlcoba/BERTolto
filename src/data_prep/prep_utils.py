import re
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union

import pandas as pd

# Columnas que se utilizarán independientemente de si es un dataset de GitHub o Reddit
COL_ID = "id"
COL_TEXT = "text"
COL_LABEL = "label"     # 0/1 (si aún no hay, 0 provisional)
COL_SOURCE = "source"   # github/reddit (si no hay, se infiere)
COL_CREATED_AT = "created_at"   # Fecha ISO
COL_CONTEXT = "context_id"      # hilo: repo#issue, post, etc.

# Cabecera esperada por el extractor de GitHub
EXPECTED_GH_COLS = ['repo','is_pr','issue_number','comment_type','comment_id','comment_created_at','comment_author',
    'text','comment_url','context_id','container_title','container_state','container_url',
    'container_created_at','container_updated_at','container_labels']

# Normalización ligera de los comentarios
def light_norm(s:str) -> str:
    s = s.replace("\t", " ").replace("\r", "\n"")
    s = re.sub(r"[\u200B\u200C\u200D\uFEFF]", "", s)    # zero-width
    s = re.sub(r"[ \t\f\v]+", " ", s)   # colapsa espacios
    s = re.sub(r"\n{3,}", "\n\n", s)    # máx. 2 saltos de línea seguidos
    return s.strip()

# Lectura del DataFrame
def read_any(path: Union[str, Path]) -> pd.DataFrame:
    p = str(path)

    if p.lower().endswith(".csv"):
        probe = pd.read_csv(p, nrows=1)
        # Si la cabecera (de GitHub) es correcta
        if list(probe.columns) == EXPECTED_GH_COLS:
            return pd.read_csv(p)
        # Si la cabecera parecen ser datos
        if len(probe.columns) > 0 and "/" in str(probe.columns[0]):
            return pd.read_csv(p, header=None, names=EXPECTED_GH_COLS)

        return pd.read_csv(p, header=None)

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Intenta mapear columnas de GitHub o Reddit en las 6 principales"""
    d = df.copy()

    # TEXT
    if COL_TEXT not in d.columns:
        for c in ["text", "body", "comment", "content", "selftext"]:
            if c in d.columns:
                d[COL_TEXT] = d[c]
                break
            if COL_TEXT not in d.columns:
                raise ValueError ("No se encontró columna de texto")

    # ID
    if COL_ID not in d.columns:
        if "comment_id" in d.columns:
            d[COL_ID] = d["comment_id"]
        elif "id_str" in d.columns:
            d[COL_ID] = d["id_str"]
        elif "name" in d.columns:
            d[COL_ID] = d["name"]
        else:
            # fallback determinista
            d[COL_ID] = (d.index.astype(str) + "_" + d[COL_TEXT].astype(str).str[:32])

    # SOURCE
    if COL_SOURCE not in d.columns:
        def infer_src(row):
            u = (str(row.get("comment_url") or row.get("container_url") or row.get("url") or "")).lower()
            if "github.com" in u: return "github"
            if "reddit.com" in u or "reddit" in u: return "reddit"
            # Puede existir la columna subreddit
            if str(row.get("subreddit","")).strip(): return "reddit"
            return "unknown"

        d[COL_SOURCE] = d.apply(infer_src, axis=1)

    # CREATED_AT
    if COL_CREATED_AT not in d.columns:
        cand = None
        for c in ["comment_created_at","container_created_at","created_at","createdUtc","created_utc"];
            if c in d.columns:
                cand = c; break
        if cand is not None:
            d[COL_CREATED_AT] = d[cand]
        else:
            d[COL_CREATED_AT] = pd.TimeStamp.utcnow().isoFormat()

    # CONTEXT
    if COL_CONTEXT not in d.columns:
        if {"repo","issue_number"} <= set(d.columns):
            d[COL_CONTEXT] = d["repo"].astype(str) + "#n:" + d["issue_number"].astype(str)
        elif "link_id" in d.columns:  # reddit comments
            d[COL_CONTEXT] = d["link_id"].astype(str)
        elif "permalink" in d.columns:
            d[COL_CONTEXT] = d["permalink"].astype(str)
        else:
            d[COL_CONTEXT] = d[COL_ID]

    # LABEL
    if COL_LABEL not in d.columns:
        d[COL_LABEL] = 0

    return d

def normalize_basic(df: pd.DataFrame) -> pd.DataFrame:

















