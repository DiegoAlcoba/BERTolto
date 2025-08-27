#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import json
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import praw
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
# Configuración por defecto (puedes sobreescribir via CLI o .env)
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()  # lee .env del CWD si existe

DEFAULT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "vuln-collector/1.0")
DEFAULT_DAYS = int(os.getenv("REDDIT_DIAS_ATRAS", "365"))
DEFAULT_OUT_CSV = Path(os.getenv("REDDIT_OUT_CSV", "../../../data/comentarios_reddit_raw_lastyear.csv"))
DEFAULT_CHECKPOINT_DIR = Path(os.getenv("REDDIT_CHECKPOINT_DIR", "../../../data/checkpoints/reddit"))
DEFAULT_SLEEP_PER_SUBMISSION = float(os.getenv("REDDIT_SLEEP_PER_SUBMISSION", "0.0"))

CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────────────────────
def iso_from_epoch(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def read_subs_file(path: Path) -> List[str]:
    subs: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            subs.append(s)
    return subs


def ckpt_path_for(sub: str, ckpt_dir: Path) -> Path:
    return ckpt_dir / f"{sub}.json"


def load_checkpoint(sub: str, ckpt_dir: Path) -> Optional[float]:
    """Devuelve el timestamp (epoch) de la publicación más antigua ya procesada, o None si no hay checkpoint."""
    p = ckpt_path_for(sub, ckpt_dir)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        ts = data.get("oldest_processed_submission_created_utc")
        if isinstance(ts, (int, float)):
            return float(ts)
    except Exception:
        pass
    return None


def save_checkpoint(sub: str, ckpt_dir: Path, oldest_ts: float) -> None:
    ensure_parent(ckpt_dir / "dummy.txt")
    data = {
        "subreddit": sub,
        "oldest_processed_submission_created_utc": float(oldest_ts),
        "oldest_processed_submission_iso": iso_from_epoch(oldest_ts),
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    ckpt_path_for(sub, ckpt_dir).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def init_reddit(user_agent: str) -> Optional[praw.Reddit]:
    if not all([CLIENT_ID, CLIENT_SECRET, user_agent]):
        print("Error: define REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET y REDDIT_USER_AGENT (p. ej. en .env).")
        return None
    return praw.Reddit(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, user_agent=user_agent)


# ──────────────────────────────────────────────────────────────────────────────
# Extracción
# ──────────────────────────────────────────────────────────────────────────────
def extract_last_n_days_to_csv(
    reddit: praw.Reddit,
    subreddits: List[str],
    out_csv: Path,
    days_back: int,
    ckpt_dir: Path,
    sleep_per_submission: float = 0.0,
    append: bool = True,
) -> None:
    ensure_parent(out_csv)
    ensure_parent(ckpt_dir / "dummy.txt")

    start_dt = datetime.now(timezone.utc) - timedelta(days=days_back)
    start_epoch = start_dt.timestamp()

    # Abrimos CSV (append o write)
    write_header = True
    if append and out_csv.exists():
        write_header = False

    total_written = 0
    print(f"Umbral temporal: publicaciones con created_utc >= {start_dt.strftime('%Y-%m-%d')} UTC")
    print(f"Salida CSV: {out_csv.resolve()}")
    print(f"Checkpoint dir: {ckpt_dir.resolve()}")

    with out_csv.open("a" if append else "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
        # CSV con al menos: comment_id, body, author, subreddit, created_utc, permalink, link_id, parent_id
                "subreddit",
                "submission_id",
                "submission_created_utc",
                "submission_title",
                "comment_id",
                "comment_created_utc",
                "comment_author",
                "comment_body",
                "comment_permalink",
                "submission_url",
            ])

        for sub in subreddits:
            try:
                sr = reddit.subreddit(sub)
                print(f"\n--- r/{sub} ---")
                ckpt_ts = load_checkpoint(sub, ckpt_dir)  # más antigua ya procesada
                if ckpt_ts is not None and ckpt_ts <= start_epoch:
                    print("  · Ventana ya cubierta previamente (checkpoint ≤ umbral). Nada que hacer.")
                    continue

                processed_any = False
                oldest_ts_this_run: Optional[float] = None
                seen_submissions = 0
                written_sub_comments = 0

                # new() devuelve descendente por fecha (más nuevas primero)
                for submission in sr.new(limit=None):
                    seen_submissions += 1

                    # 1) Si tenemos checkpoint y esta submission es más reciente (>= ckpt_ts), ya fue procesada → saltar
                    if ckpt_ts is not None and submission.created_utc >= ckpt_ts:
                        continue

                    # 2) Si es anterior a la ventana, cortamos (hemos llegado a >N días atrás)
                    if submission.created_utc < start_epoch:
                        print("  · Corte por fecha alcanzado. Paro este subreddit.")
                        break

                    # Procesar comentarios de esta publicación
                    try:
                        submission.comments.replace_more(limit=None)
                    except Exception as e:
                        print(f"  ! replace_more() falló en {submission.id}: {e}. Continúo.")
                        continue

                    wrote_this_submission = False
                    for c in submission.comments.list():
                        try:
                            writer.writerow([
                                str(sr.display_name),
                                submission.id,
                                iso_from_epoch(submission.created_utc),
                                submission.title,
                                c.id,
                                iso_from_epoch(getattr(c, "created_utc", submission.created_utc)),
                                (c.author.name if c.author else None),
                                c.body,  # texto puro
                                f"https://www.reddit.com{getattr(c, 'permalink', '')}",
                                f"https://www.reddit.com{submission.permalink}",
                            ])
                            written_sub_comments += 1
                            total_written += 1
                            wrote_this_submission = True
                        except Exception as e:
                            print(f"  ! Error escribiendo comentario {getattr(c, 'id', '?')}: {e}")

                    # Si escribimos comentarios de esta submission, actualizamos el oldest_ts_this_run
                    if wrote_this_submission:
                        processed_any = True
                        if oldest_ts_this_run is None or submission.created_utc < oldest_ts_this_run:
                            oldest_ts_this_run = submission.created_utc

                    if sleep_per_submission > 0:
                        time.sleep(sleep_per_submission)

                print(f"  · Submissions vistas: {seen_submissions}, comentarios escritos: {written_sub_comments}")

                # Guardar checkpoint si procesamos algo nuevo
                if processed_any and oldest_ts_this_run is not None:
                    save_checkpoint(sub, ckpt_dir, oldest_ts_this_run)
                    print(f"  · Checkpoint actualizado: oldest_submission={iso_from_epoch(oldest_ts_this_run)}")

            except Exception as e:
                print(f"!! Error en r/{sub}: {e}. Continúo con el siguiente.")

    print(f"\n¡Listo! Comentarios escritos: {total_written}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    import argparse

    ap = argparse.ArgumentParser(description="Extrae comentarios PUROS de subreddits.txt (últimos N días) con checkpoint por subreddit.")
    ap.add_argument("--subs-file", type=str, required=False, help="Ruta a .txt con un subreddit por línea (sin 'r/').")
    ap.add_argument("--out-csv", type=str, default=str(DEFAULT_OUT_CSV), help="Ruta de salida CSV (se crea si no existe).")
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS, help="Días hacia atrás (ventana). Por defecto 365.")
    ap.add_argument("--checkpoint-dir", type=str, default=str(DEFAULT_CHECKPOINT_DIR), help="Directorio para archivos de checkpoint por subreddit.")
    ap.add_argument("--sleep-per-submission", type=float, default=DEFAULT_SLEEP_PER_SUBMISSION, help="Pausa (seg) entre publicaciones para ser cortés con la API.")
    ap.add_argument("--no-append", action="store_true", help="No anexar: sobrescribe el CSV (escribe cabecera de nuevo).")
    ap.add_argument("--user-agent", type=str, default=DEFAULT_USER_AGENT, help="User-Agent para PRAW.")

    args = ap.parse_args()

    # Subreddits: desde --subs-file o variable de entorno REDDIT_SUBS (coma-separado) o lista por defecto
    subs: List[str] = []
    if args.subs_file:
        subs = read_subs_file(Path(args.subs_file))
    else:
        env_subs = os.getenv("REDDIT_SUBS")
        if env_subs:
            subs = [s.strip() for s in env_subs.split(",") if s.strip()]
        else:
            subs = [
                "netsec", "AskNetsec", "cybersecurity", "ReverseEngineering",
                "exploitdev", "websecurity", "websecurityresearch", "Malware", "netsecstudents"
            ]

    if not subs:
        print("No hay subreddits.txt para procesar. Usa --subs-file o REDDIT_SUBS.")
        raise SystemExit(1)

    reddit = init_reddit(args.user_agent)
    if not reddit:
        raise SystemExit(1)

    extract_last_n_days_to_csv(
        reddit=reddit,
        subreddits=subs,
        out_csv=Path(args.out_csv),
        days_back=args.days,
        ckpt_dir=Path(args.checkpoint_dir),
        sleep_per_submission=args.sleep_per_submission,
        append=not args.no_append,
    )


if __name__ == "__main__":
    main()
