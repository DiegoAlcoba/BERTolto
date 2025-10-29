#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extractor de comentarios de Reddit con checkpoints, ventanas flexibles, filtro por título/body
y selección del comentario top-voted por publicación.

- Guarda CSV incremental y <out-base>.state.json con newest/oldest vistos por subreddit.
- Filtra SIEMPRE por ventana temporal.
- Palabras clave: en título y/o body del post (configurable).
- Límite de comentarios por post (por defecto 1 → top-voted). Opción de incluir el body del post.

Modos de ventana:
    * days   : desde hoy hasta N días atrás (--days)
    * newer  : desde el más moderno ya guardado hasta hoy
    * older  : desde N días antes del más antiguo ya guardado hasta ese "más antiguo" (--days)
    * range  : de --from-days a --to-days (ambos relativos a hoy; p.ej. 30→7)
    * window : por fechas ISO --since / --until

Requisitos de entorno:
  REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT (o pásalo por CLI)

Ejemplo:
  python reddit_scrapper.py --subs-file subs.txt \
    --out-base ../../../data/reddit_comments/training/reddit_2023_now \
    --mode days --days 365 \
    --include-post-body \
    --max-comments-per-post 1 \
    --use-default-title-keywords --use-default-body-keywords
"""

import os
import sys
import csv
import json
import time
import random
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple, Dict, Any

import praw
import prawcore
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────────────
# Carga .env (opcional)
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()

# ──────────────────────────────────────────────────────────────────────────────
# Defaults y palabras clave
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "vuln-collector/1.0")

DEFAULT_TITLE_KEYWORDS = [
    # genéricos bug/error/fallo
    "bug", "error", "failure", "fault", "issue", "regression", "crash", "hang", "freeze",
    "segfault", "panic", "unexpected", "incorrect", "wrong", "mismatch",
    "null pointer", "nil pointer", "undefined behavior", "ub", "race condition",
    "deadlock", "livelock", "timeout", "latency spike",
    "memory leak", "leak", "oom", "out of memory", "high memory", "use-after-free",
    "double free", "dangling", "buffer overflow", "stack overflow", "heap overflow",
    "out-of-bounds", "oob", "off-by-one", "overread", "overwrite", "corruption", "corrupt",
    "infinite loop", "loop", "recursion", "stack exhaustion",
    # seguridad
    "security", "vulnerability", "vuln", "cve", "exploit", "rce", "lpe",
    "privilege escalation", "xss", "cross-site scripting", "csrf", "ssrf", "sqli",
    "sql injection", "command injection", "code injection", "template injection",
    "path traversal", "directory traversal", "insecure", "exposure",
    "information disclosure", "auth bypass", "authentication bypass",
    "authorization bypass", "weak cryptography", "weak crypto", "insecure default",
    "hardcoded secret", "sensitive data", "token leak", "credential leak",
]

# Para body puedes usar la misma lista o ampliarla con términos de síntomas
DEFAULT_BODY_KEYWORDS = DEFAULT_TITLE_KEYWORDS + [
    "se cuelga", "se bloquea", "comportamiento extraño", "no esperado",
    "datos sensibles", "credenciales", "token", "clave", "contraseña",
]

# ──────────────────────────────────────────────────────────────────────────────
# Utilidades
# ──────────────────────────────────────────────────────────────────────────────
def iso_from_epoch(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def epoch_from_iso(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()

def parse_iso_optional(s: Optional[str]) -> Optional[float]:
    if not s:
        return None
    return epoch_from_iso(s)

def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def read_list_file(path: Path) -> List[str]:
    out: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out

def normalize_keywords(seq: List[str]) -> List[str]:
    out = []
    for s in seq:
        ss = s.strip().lower()
        if ss:
            out.append(ss)
    return out

def matches(text: Optional[str], patterns: List[str]) -> bool:
    if not patterns:
        return True
    t = (text or "").lower()
    return any(p in t for p in patterns)

def init_reddit(user_agent: str) -> praw.Reddit:
    cid = os.getenv("REDDIT_CLIENT_ID")
    csec = os.getenv("REDDIT_CLIENT_SECRET")
    if not all([cid, csec, user_agent]):
        print("Error: define REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET y REDDIT_USER_AGENT (.env o CLI).")
        sys.exit(1)
    return praw.Reddit(client_id=cid, client_secret=csec, user_agent=user_agent)

# ──────────────────────────────────────────────────────────────────────────────
# Estado y dedupe
# ──────────────────────────────────────────────────────────────────────────────
def load_state(path: Path) -> Dict[str, Dict[str, float]]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def save_state(path: Path, state: Dict[str, Dict[str, float]]) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

class Writer:
    def __init__(self, csv_path: Path, on_write=None):
        self.path = csv_path
        ensure_parent(csv_path)
        self._csv_new = not csv_path.exists()
        self._fp = csv_path.open("a", newline="", encoding="utf-8")
        self._w = csv.writer(self._fp)
        self._on_write = on_write
        if self._csv_new:
            self._w.writerow([
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
        # dedupe por comment_id
        self.seen = set()
        try:
            with self.path.open("r", encoding="utf-8") as f:
                r = csv.reader(f)
                header = next(r, None)
                idx = header.index("comment_id") if header and "comment_id" in header else 4
                for row in r:
                    if row:
                        self.seen.add(row[idx])
        except Exception:
            pass

    def write_row(self, row: Dict[str, Any]):
        self._w.writerow([
            row["subreddit"],
            row["submission_id"],
            row["submission_created_utc"],
            row["submission_title"],
            row["comment_id"],
            row["comment_created_utc"],
            row["comment_author"],
            row["comment_body"],
            row["comment_permalink"],
            row["submission_url"],
        ])
        if self._on_write:
            try:
                self._on_write(row)
            except Exception:
                pass

    def write_unique(self, row: Dict[str, Any]) -> bool:
        rid = row["comment_id"]
        if rid in self.seen:
            return False
        self.write_row(row)
        self.seen.add(rid)
        return True

    def flush(self):
        self._fp.flush()

    def close(self):
        self._fp.close()

# ──────────────────────────────────────────────────────────────────────────────
# Ventanas
# ──────────────────────────────────────────────────────────────────────────────
def compute_window(mode: str,
                   days: Optional[int],
                   since: Optional[str],
                   until: Optional[str],
                   from_days: Optional[int],
                   to_days: Optional[int],
                   sub: str,
                   state: Dict[str, Dict[str, float]]) -> Tuple[float, float]:
    now = datetime.now(timezone.utc).timestamp()
    eps = 1.0
    s_since = parse_iso_optional(since)
    s_until = parse_iso_optional(until)
    rep = state.get(sub, {})
    newest = rep.get("newest_comment_ts")
    oldest = rep.get("oldest_comment_ts")

    if mode == "days":
        d = days if days is not None else 365
        return now - d * 86400, now

    if mode == "newer":
        lo = (newest + eps) if (newest is not None) else (now - (days or 365) * 86400)
        hi = s_until if s_until is not None else now
        return lo, hi

    if mode == "older":
        hi = (oldest - eps) if (oldest is not None) else now
        d = days if days is not None else 365
        lo = hi - d * 86400
        if lo > hi:
            lo = hi - 1
        return lo, hi

    if mode == "range":
        if from_days is None or to_days is None:
            raise ValueError("--mode range requiere --from-days y --to-days")
        if from_days < to_days:
            raise ValueError("--from-days debe ser >= --to-days (ej. 30 y 7)")
        lo = now - from_days * 86400
        hi = now - to_days * 86400
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi

    if mode == "window":
        hi = s_until if s_until is not None else now
        lo = s_since if s_since is not None else (hi - (days or 365) * 86400)
        if lo > hi:
            lo, hi = hi - 1, hi
        return lo, hi

    raise ValueError(f"Modo no soportado: {mode}")

# ──────────────────────────────────────────────────────────────────────────────
# Rate limit y backoff
# ──────────────────────────────────────────────────────────────────────────────
def sleep_with_jitter(seconds: float):
    time.sleep(seconds + random.uniform(0.0, min(1.0, seconds * 0.1)))

def handle_rate_limit(e: Exception, fallback_wait: float = 30.0):
    # prawcore.TooManyRequests expone headers con Retry-After normalmente
    wait = fallback_wait
    if isinstance(e, prawcore.exceptions.TooManyRequests):
        try:
            ra = e.response.headers.get("Retry-After")
            if ra:
                wait = max(wait, float(ra))
        except Exception:
            pass
    print(f"[rate-limit] Cuota agotada. Espero {wait:.1f}s…")
    sleep_with_jitter(wait)

# ──────────────────────────────────────────────────────────────────────────────
# Núcleo de extracción
# ──────────────────────────────────────────────────────────────────────────────
def row_from_submission_body(sr_name: str, subm) -> Dict[str, Any]:
    # Representamos el body como "comentario" sintético
    return {
        "subreddit": sr_name,
        "submission_id": subm.id,
        "submission_created_utc": iso_from_epoch(subm.created_utc),
        "submission_title": subm.title,
        "comment_id": f"reddit_postbody_{subm.id}",
        "comment_created_utc": iso_from_epoch(subm.created_utc),
        "comment_author": (subm.author.name if subm.author else None),
        "comment_body": (subm.selftext or ""),
        "comment_permalink": f"https://www.reddit.com{subm.permalink}",
        "submission_url": f"https://www.reddit.com{subm.permalink}",
    }

def row_from_comment(sr_name: str, subm, c) -> Dict[str, Any]:
    return {
        "subreddit": sr_name,
        "submission_id": subm.id,
        "submission_created_utc": iso_from_epoch(subm.created_utc),
        "submission_title": subm.title,
        "comment_id": c.id,
        "comment_created_utc": iso_from_epoch(getattr(c, "created_utc", subm.created_utc)),
        "comment_author": (c.author.name if c.author else None),
        "comment_body": (c.body or ""),
        "comment_permalink": f"https://www.reddit.com{getattr(c, 'permalink', subm.permalink)}",
        "submission_url": f"https://www.reddit.com{subm.permalink}",
    }

def extract_for_subreddit(
    reddit: praw.Reddit,
    sub: str,
    writer: Writer,
    state: Dict[str, Dict[str, float]],
    low_ts: float,
    high_ts: float,
    include_post_body: bool,
    max_comments_per_post: Optional[int],
    title_keywords: List[str],
    body_keywords: List[str],
) -> int:
    """
    - Filtra por ventana usando created_utc del *post* (eficiente).
    - Aplica filtro de keywords en título y/o body según se hayan pasado.
    - Si max_comments_per_post=1 → escoge el comentario más votado (top-level).
    - Actualiza state[newest/oldest] con timestamps de los comentarios que se escriben.
    """
    sr = reddit.subreddit(sub)
    wrote = 0

    rep_entry = state.setdefault(sub, {})
    newest = rep_entry.get("newest_comment_ts")
    oldest = rep_entry.get("oldest_comment_ts")

    def update_minmax(ts_epoch: float):
        nonlocal newest, oldest
        if newest is None or ts_epoch > newest:
            newest = ts_epoch
        if oldest is None or ts_epoch < oldest:
            oldest = ts_epoch
        rep_entry["newest_comment_ts"] = newest
        rep_entry["oldest_comment_ts"] = oldest

    # Recorremos por new() (descendente). Cortamos cuando sobrepasamos low_ts
    seen_posts = 0
    while True:
        try:
            for subm in sr.new(limit=None):
                seen_posts += 1
                # ventana por post
                if subm.created_utc > high_ts:
                    # Post demasiado reciente; seguimos
                    pass
                elif subm.created_utc < low_ts:
                    # Llegamos al corte inferior
                    print(f"  · Corte por fecha en r/{sub}: {iso_from_epoch(subm.created_utc)} < {iso_from_epoch(low_ts)}")
                    raise StopIteration

                # Filtro por keywords
                if title_keywords and not matches(subm.title, title_keywords):
                    continue
                if body_keywords and not matches(getattr(subm, "selftext", ""), body_keywords):
                    continue

                # Body del post (opcional)
                if include_post_body and (subm.selftext or "").strip():
                    row = row_from_submission_body(str(sr.display_name), subm)
                    if writer.write_unique(row):
                        wrote += 1
                        update_minmax(subm.created_utc)

                # Comentarios
                if (max_comments_per_post is not None) and (max_comments_per_post <= 0):
                    continue

                # Cargar todo el bosque de comentarios una vez
                # (para top-voted basta con top-level; aun así, cargar y elegir)
                try:
                    subm.comments.replace_more(limit=0)
                except prawcore.exceptions.TooManyRequests as e:
                    handle_rate_limit(e, fallback_wait=60.0)
                    # reintento único de replace_more
                    try:
                        subm.comments.replace_more(limit=0)
                    except Exception as e2:
                        print(f"  ! replace_more falló en {subm.id}: {e2}. Continúo.")
                        continue
                except Exception as e:
                    print(f"  ! replace_more falló en {subm.id}: {e}. Continúo.")
                    continue

                # Top-level comments (CommentForest es iterable)
                top_level = [c for c in list(subm.comments) if getattr(c, "parent_id", "").startswith("t3_")]
                # Filtramos removidos/borrados
                filtered = [c for c in top_level if getattr(c, "body", "").strip() not in ("[removed]", "[deleted]")]
                # Orden por score desc
                filtered.sort(key=lambda c: getattr(c, "score", 0), reverse=True)

                count = 0
                for c in filtered:
                    # Aplicamos ventana por *comentario* para ser coherentes con GH
                    cts = getattr(c, "created_utc", subm.created_utc)
                    if not (low_ts <= cts <= high_ts):
                        continue

                    row = row_from_comment(str(sr.display_name), subm, c)
                    if writer.write_unique(row):
                        wrote += 1
                        count += 1
                        update_minmax(cts)
                    if (max_comments_per_post is not None) and (count >= max_comments_per_post):
                        break
            # Si agotamos el iterador sin StopIteration, no hay más (API pagina internamente).
            break

        except prawcore.exceptions.TooManyRequests as e:
            handle_rate_limit(e, fallback_wait=60.0)
            continue
        except StopIteration:
            break
        except (prawcore.exceptions.ServerError,
                prawcore.exceptions.ResponseException,
                prawcore.exceptions.RequestException) as e:
            # backoff transitorio
            wait = min(30.0, 5.0 + random.uniform(0, 3))
            print(f"[transient] {type(e).__name__}: {e}. Reintento en {wait:.1f}s…")
            sleep_with_jitter(wait)
            continue
        except Exception as e:
            print(f"!! Error en r/{sub}: {e}. Continúo.")
            continue

    print(f"  · r/{sub}: posts vistos={seen_posts}, filas escritas={wrote}")
    return wrote

# ──────────────────────────────────────────────────────────────────────────────
# Main / CLI
# ──────────────────────────────────────────────────────────────────────────────
def main():
    import argparse

    ap = argparse.ArgumentParser(description="Extractor Reddit con checkpoints, ventanas y top-voted comments.")
    ap.add_argument("--subs-file", type=str, help="Ruta a .txt con un subreddit por línea (sin 'r/').")
    ap.add_argument("--out-base", type=str, default="../../data/reddit_comments/training/rd_comments_2023_now",
                    help="Ruta base sin extensión (.csv y .state.json). Ej: data/reddit_2023_now")
    ap.add_argument("--user-agent", type=str, default=DEFAULT_USER_AGENT, help="Reddit user-agent.")

    # Ventanas
    ap.add_argument("--mode", type=str, choices=["days", "newer", "older", "range", "window"], default="days",
                    help="days|newer|older|range|window (como en GH).")
    ap.add_argument("--days", type=int, default=None, help="Para 'days' y 'older'; fallback en 'newer/window'.")
    ap.add_argument("--from-days", type=int, default=None, help="Range: inicio relativo.")
    ap.add_argument("--to-days", type=int, default=None, help="Range: fin relativo.")
    ap.add_argument("--since", type=str, default=None, help="Window: ISO inferior.")
    ap.add_argument("--until", type=str, default=None, help="Window/Newer: ISO superior.")

    # Filtros por keywords
    ap.add_argument("--use-default-title-keywords", action="store_true", help="Activa defaults ES/EN para título.")
    ap.add_argument("--use-default-body-keywords", action="store_true", help="Activa defaults ES/EN para body.")
    ap.add_argument("--title-keywords", type=str, default=None, help="Lista coma-separada para título.")
    ap.add_argument("--body-keywords", type=str, default=None, help="Lista coma-separada para body.")
    ap.add_argument("--title-keywords-file", type=str, default=None, help="Archivo con una palabra/frase por línea.")
    ap.add_argument("--body-keywords-file", type=str, default=None, help="Archivo con una palabra/frase por línea.")

    # Qué guardar por post
    ap.add_argument("--include-post-body", action="store_true", help="Guarda también el body del post como fila.")
    ap.add_argument("--max-comments-per-post", type=int, default=1,
                    help="Máximo de comentarios por post (1 = top-voted; 0 = ninguno; >1 = top-N).")

    args = ap.parse_args()

    #Comando:python reddit_extractor_checkpointed.py --mode window --since 2023-01-01T00:00:00Z --use-default-title-keywords --include-post-body --max-comments-per-post 1

    # Subreddits desde archivo, env o defaults
    subs: List[str] = []
    if args.subs_file:
        subs = read_list_file(Path(args.subs_file))
    else:
        env_subs = os.getenv("REDDIT_SUBS")
        if env_subs:
            subs = [s.strip() for s in env_subs.split(",") if s.strip()]
        else:
            subs = ["netsec", "AskNetsec", "cybersecurity", "ReverseEngineering",
                    "exploitdev", "websecurity", "websecurityresearch", "Malware", "netsecstudents"]
    if not subs:
        print("No hay subreddits para procesar.")
        sys.exit(1)

    # Preparar keywords
    title_kw: List[str] = []
    body_kw: List[str] = []

    if args.use_default_title_keywords:
        title_kw.extend(DEFAULT_TITLE_KEYWORDS)
    if args.use_default_body_keywords:
        body_kw.extend(DEFAULT_BODY_KEYWORDS)

    if args.title_keywords:
        title_kw.extend([x.strip() for x in args.title_keywords.split(",") if x.strip()])
    if args.body_keywords:
        body_kw.extend([x.strip() for x in args.body_keywords.split(",") if x.strip()])

    if args.title_keywords_file:
        fp = Path(args.title_keywords_file)
        if fp.exists():
            title_kw.extend(read_list_file(fp))
        else:
            print(f"[warn] No existe --title-keywords-file: {fp}")

    if args.body_keywords_file:
        fp = Path(args.body_keywords_file)
        if fp.exists():
            body_kw.extend(read_list_file(fp))
        else:
            print(f"[warn] No existe --body-keywords-file: {fp}")

    title_kw = normalize_keywords(title_kw)
    body_kw = normalize_keywords(body_kw)

    if title_kw:
        print(f"[info] Filtro TÍTULO activo ({len(title_kw)} términos).")
    if body_kw:
        print(f"[info] Filtro BODY activo ({len(body_kw)} términos).")

    # Reddit client
    reddit = init_reddit(args.user_agent)

    # Rutas
    base = Path(args.out_base)
    csv_path = base.with_suffix(".csv")
    state_path = base.with_suffix(".state.json")

    # Estado + writer
    state = load_state(state_path)

    # Hook para actualizar min/max desde el writer (ya lo manejamos en extractor; aquí no necesario)
    writer = Writer(csv_path)

    total = 0
    for sub in subs:
        low_ts, high_ts = compute_window(
            mode=args.mode, days=args.days, since=args.since, until=args.until,
            from_days=args.from_days, to_days=args.to_days, sub=sub, state=state
        )
        print(f"\n==> r/{sub} | modo={args.mode} | ventana [{iso_from_epoch(low_ts)} .. {iso_from_epoch(high_ts)}]")

        try:
            n = extract_for_subreddit(
                reddit=reddit,
                sub=sub,
                writer=writer,
                state=state,
                low_ts=low_ts,
                high_ts=high_ts,
                include_post_body=args.include_post_body,
                max_comments_per_post=args.max_comments_per_post,
                title_keywords=title_kw,
                body_keywords=body_kw,
            )
            total += n
            writer.flush()
        except Exception as e:
            print(f"  ! Error en r/{sub}: {e}. Continúo.")
            continue

    writer.close()
    save_state(state_path, state)
    print(f"\nExtracción completada. Filas nuevas: {total}")
    print(f"CSV:   {csv_path}")
    print(f"STATE: {state_path}")

if __name__ == "__main__":
    main()
