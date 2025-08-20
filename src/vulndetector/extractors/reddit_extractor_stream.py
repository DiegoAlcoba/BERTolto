#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import json
import time
import signal
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional

import praw
from dotenv import load_dotenv

# ─────────────────────────────
# Config y defaults (.env opcional)
# ─────────────────────────────
load_dotenv()

CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
USER_AGENT = os.getenv("REDDIT_USER_AGENT", "vuln-collector/stream/1.0")

DEFAULT_OUT = Path(os.getenv("REDDIT_OUT_PATH", "../../../data/reddit_stream"))
DEFAULT_FMT = os.getenv("REDDIT_OUT_FORMAT", "csv")  # csv | jsonl
DEFAULT_CHECKPOINT_DIR = Path(os.getenv("REDDIT_CHECKPOINT_DIR", "../../../data/checkpoints/reddit_stream"))
DEFAULT_SUBS = [s.strip() for s in os.getenv("REDDIT_SUBS", "").split(",") if s.strip()]
DEFAULT_CATCHUP_LIMIT = int(os.getenv("REDDIT_CATCHUP_LIMIT", "1000"))  # comentarios recientes por subreddit
CHECKPOINT_FLUSH_EVERY = int(os.getenv("REDDIT_CKPT_FLUSH_EVERY", "50"))  # flush tras N comentarios escritos
SLEEP_EMPTY_LOOP = float(os.getenv("REDDIT_SLEEP_EMPTY", "1.0"))  # espera cuando no hay nuevos

# ─────────────────────────────
# Helpers
# ─────────────────────────────
def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def iso_from_epoch(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def load_checkpoint(ckpt_dir: Path) -> Dict[str, Dict[str, Optional[str]]]:
    """Devuelve {subreddit: {'last_ts': float|None, 'last_id': str|None}}"""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Dict[str, Optional[str]]] = {}
    for p in ckpt_dir.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            sub = data.get("subreddit") or p.stem
            out[sub] = {
                "last_ts": data.get("last_comment_created_utc"),
                "last_id": data.get("last_comment_id"),
            }
        except Exception:
            continue
    return out

def save_checkpoint_for(sub: str, ckpt_dir: Path, last_ts: float, last_id: str) -> None:
    ensure_parent(ckpt_dir / "x")
    data = {
        "subreddit": sub,
        "last_comment_created_utc": float(last_ts),
        "last_comment_created_iso": iso_from_epoch(last_ts),
        "last_comment_id": last_id,
        "updated_at": iso_from_epoch(time.time()),
    }
    (ckpt_dir / f"{sub}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def init_reddit() -> Optional[praw.Reddit]:
    if not all([CLIENT_ID, CLIENT_SECRET, USER_AGENT]):
        print("ERROR: define REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET y REDDIT_USER_AGENT (p.ej. en .env).")
        return None
    return praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=USER_AGENT)

def read_subs_file(path: Optional[str]) -> List[str]:
    if not path:
        return DEFAULT_SUBS
    subs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                subs.append(s)
    return subs

# ─────────────────────────────
# Salida (CSV/JSONL)
# ─────────────────────────────
class Writer:
    def __init__(self, base_path: Path, fmt: str = "csv"):
        self.fmt = fmt.lower()
        if self.fmt not in ("csv", "jsonl"):
            raise ValueError("Formato no soportado. Usa csv o jsonl.")
        self.path = base_path.with_suffix("." + self.fmt)
        ensure_parent(self.path)
        self._opened = False
        self._fh = None
        self._csv_writer = None
        self._write_header_if_needed()

    def _write_header_if_needed(self):
        if self.fmt == "csv":
            new_file = not self.path.exists()
            self._fh = open(self.path, "a", newline="", encoding="utf-8")
            self._opened = True
            self._csv_writer = csv.writer(self._fh)
            if new_file:
                self._csv_writer.writerow([
                    "subreddit",
                    "comment_id",
                    "comment_created_utc",
                    "comment_created_iso",
                    "comment_author",
                    "comment_body",
                    "comment_permalink",
                    "submission_id",
                    "submission_created_utc",
                    "submission_created_iso",
                    "submission_title",
                    "submission_url",
                ])
        else:
            self._fh = open(self.path, "a", encoding="utf-8")
            self._opened = True

    def write(self, rec: dict):
        if self.fmt == "csv":
            self._csv_writer.writerow([
                rec.get("subreddit"),
                rec.get("comment_id"),
                rec.get("comment_created_utc"),
                rec.get("comment_created_iso"),
                rec.get("comment_author"),
                rec.get("comment_body"),
                rec.get("comment_permalink"),
                rec.get("submission_id"),
                rec.get("submission_created_utc"),
                rec.get("submission_created_iso"),
                rec.get("submission_title"),
                rec.get("submission_url"),
            ])
        else:
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def flush(self):
        if self._opened:
            self._fh.flush()

    def close(self):
        if self._opened:
            self._fh.close()
            self._opened = False

# ─────────────────────────────
# Catch-up (al arrancar) + streaming
# ─────────────────────────────
def do_catchup(reddit: praw.Reddit, subs: List[str], ckpt: Dict[str, Dict[str, Optional[str]]],
               writer: Writer, limit_per_sub: int) -> int:
    """
    Recupera comentarios recientes por subreddit usando subreddit.comments(limit=N)
    y filtra los que son > checkpoint. Es mejor para ventanas cortas (horas/días).
    """
    written = 0
    for sub in subs:
        last_ts = float(ckpt.get(sub, {}).get("last_ts") or 0.0)
        last_id = ckpt.get(sub, {}).get("last_id")
        try:
            sr = reddit.subreddit(sub)
            # Reddit API devuelve recientes→antiguos; los ordenamos ascendente para escribir en orden temporal
            buf = []
            for c in sr.comments(limit=limit_per_sub):
                if getattr(c, "created_utc", 0.0) > last_ts:
                    buf.append(c)
            buf.sort(key=lambda x: getattr(x, "created_utc", 0.0))
            for c in buf:
                subm = c.submission
                rec = {
                    "subreddit": str(c.subreddit),
                    "comment_id": c.id,
                    "comment_created_utc": int(c.created_utc),
                    "comment_created_iso": iso_from_epoch(c.created_utc),
                    "comment_author": (c.author.name if c.author else None),
                    "comment_body": c.body,  # puro
                    "comment_permalink": f"https://www.reddit.com{getattr(c, 'permalink', '')}",
                    "submission_id": subm.id,
                    "submission_created_utc": int(subm.created_utc),
                    "submission_created_iso": iso_from_epoch(subm.created_utc),
                    "submission_title": subm.title,
                    "submission_url": f"https://www.reddit.com{subm.permalink}",
                }
                writer.write(rec)
                written += 1
                # actualizar ckpt
                ckpt.setdefault(sub, {})
                ckpt[sub]["last_ts"] = float(c.created_utc)
                ckpt[sub]["last_id"] = c.id
            # guardar ckpt del sub si hubo catchup
            if buf:
                save_checkpoint_for(sub, DEFAULT_CHECKPOINT_DIR, ckpt[sub]["last_ts"], ckpt[sub]["last_id"])
        except Exception as e:
            print(f"[catch-up] r/{sub}: {e}")
    if written:
        writer.flush()
    return written

def stream_forever(reddit: praw.Reddit, subs: List[str], ckpt: Dict[str, Dict[str, Optional[str]]], writer: Writer):
    """
    Stream en tiempo real. Filtra por checkpoint para evitar duplicados en reinicios.
    Usa pause_after para no bloquear y poder flush/checkpoint periódicamente.
    """
    multi = "+".join(subs)
    sr = reddit.subreddit(multi)
    seen = 0
    since_flush = 0

    print(f"Streaming en r/{multi} (skip_existing=True)…")
    stream = sr.stream.comments(skip_existing=True, pause_after=5)

    for item in stream:
        if item is None:
            # no hay nuevos ahora mismo
            if SLEEP_EMPTY_LOOP > 0:
                time.sleep(SLEEP_EMPTY_LOOP)
            continue
        try:
            sub = str(item.subreddit)
            last_ts = float(ckpt.get(sub, {}).get("last_ts") or 0.0)
            # si por algún motivo recibimos algo < ckpt (duplicado), lo saltamos
            if getattr(item, "created_utc", 0.0) <= last_ts:
                continue

            subm = item.submission
            rec = {
                "subreddit": sub,
                "comment_id": item.id,
                "comment_created_utc": int(item.created_utc),
                "comment_created_iso": iso_from_epoch(item.created_utc),
                "comment_author": (item.author.name if item.author else None),
                "comment_body": item.body,
                "comment_permalink": f"https://www.reddit.com{getattr(item, 'permalink', '')}",
                "submission_id": subm.id,
                "submission_created_utc": int(subm.created_utc),
                "submission_created_iso": iso_from_epoch(subm.created_utc),
                "submission_title": subm.title,
                "submission_url": f"https://www.reddit.com{subm.permalink}",
            }
            writer.write(rec)

            # actualizar ckpt en memoria
            ckpt.setdefault(sub, {})
            ckpt[sub]["last_ts"] = float(item.created_utc)
            ckpt[sub]["last_id"] = item.id

            seen += 1
            since_flush += 1
            if since_flush >= CHECKPOINT_FLUSH_EVERY:
                # flush global de ckpts (por sub)
                for s, d in ckpt.items():
                    if d.get("last_ts") and d.get("last_id"):
                        save_checkpoint_for(s, DEFAULT_CHECKPOINT_DIR, d["last_ts"], d["last_id"])
                writer.flush()
                since_flush = 0

        except Exception as e:
            print(f"[stream] error: {e}")
            time.sleep(1.0)

# ─────────────────────────────
# Main / CLI simple
# ─────────────────────────────
def main():
    import argparse

    ap = argparse.ArgumentParser(description="Stream de comentarios de Reddit con checkpoint (catch-up + tiempo real).")
    ap.add_argument("--subs-file", type=str, help="Ruta a .txt con un subreddit por línea (sin 'r/'). Alternativa: REDDIT_SUBS=netsec,AskNetsec…")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Ruta base de salida SIN extensión (se añade .csv o .jsonl)")
    ap.add_argument("--format", type=str, default=DEFAULT_FMT, choices=["csv", "jsonl"], help="Formato de salida (csv|jsonl)")
    ap.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR), help="Directorio de checkpoints por subreddit")
    ap.add_argument("--catchup-limit", type=int, default=DEFAULT_CATCHUP_LIMIT, help="Máx. comentarios recientes a revisar por subreddit al iniciar")
    args = ap.parse_args()

    subs = read_subs_file(args.subs_file)
    if not subs:
        print("No hay subreddits.txt definidos. Usa --subs-file o variable REDDIT_SUBS.")
        raise SystemExit(1)

    # preparar salidas y reddit
    out_base = Path(args.out)
    ensure_parent(out_base.with_suffix(".x"))
    global DEFAULT_CHECKPOINT_DIR
    DEFAULT_CHECKPOINT_DIR = Path(args.checkpoint_dir)

    reddit = init_reddit()
    if not reddit:
        raise SystemExit(1)

    writer = Writer(out_base, fmt=args.format)
    ckpt = load_checkpoint(DEFAULT_CHECKPOINT_DIR)

    # graceful shutdown (guardar ckpts al salir)
    def handle_sigint(sig, frame):
        print("\nSeñal recibida. Guardando checkpoints y cerrando…")
        for s, d in ckpt.items():
            if d.get("last_ts") and d.get("last_id"):
                save_checkpoint_for(s, DEFAULT_CHECKPOINT_DIR, d["last_ts"], d["last_id"])
        writer.flush()
        writer.close()
        raise SystemExit(0)
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)

    # 1) Catch-up (llenar huecos desde el último checkpoint hasta "ahora")
    wrote = do_catchup(reddit, subs, ckpt, writer, args.catchup_limit)
    if wrote:
        print(f"[catch-up] escritos {wrote} comentarios pendientes.")

    # 2) Streaming continuo (tiempo real)
    print("Entrando en modo streaming… (Ctrl+C para terminar)")
    stream_forever(reddit, subs, ckpt, writer)

if __name__ == "__main__":
    main()
