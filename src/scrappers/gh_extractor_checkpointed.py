#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extractor de comentarios de GitHub con checkpoints y ventanas flexibles + filtro por título.
- Guarda un CSV incremental y un .state.json con el comentario más nuevo y más viejo vistos por repo.
- Filtra SIEMPRE por ventana temporal (createdAt ∈ [low_ts, high_ts]).
- Opcional: filtra por palabras clave en el TÍTULO de issues/PRs (container_title).

Modos:
    * days   : desde hoy hasta N días atrás (--days)
    * newer  : desde el más moderno ya guardado hasta hoy
    * older  : desde N días antes del más antiguo ya guardado hasta ese "más antiguo" (--days)
    * range  : de --from-days a --to-days (ambos relativos a hoy; p.ej. 30→7)
    * window : por fechas ISO --since / --until

Ejemplos:
  export GITHUB_TOKEN=ghp_xxx

  # Últimos 365 días (desde hoy hacia atrás)
  python gh_extractor_checkpointed.py --repos-file repos.txt \
      --out-base data/gh_comments/lastyear --mode days --days 365 --include-review-comments

  # Incremental hacia ADELANTE desde lo último visto:
  python gh_extractor_checkpointed.py --repos-file repos.txt \
      --out-base data/gh_comments/lastyear --mode newer

  # Backfill 30 días hacia ATRÁS desde lo más viejo ya visto:
  python gh_extractor_checkpointed.py --repos-file repos.txt \
      --out-base data/gh_comments/lastyear --mode older --days 30

  # Rango relativo: de 30 días atrás a 7 días atrás (ambos incluidos)
  python gh_extractor_checkpointed.py --repos-file repos.txt \
      --out-base data/gh_comments/ranged --mode range --from-days 30 --to-days 7

  # Ventana absoluta por fechas ISO (enero de 2023):
  python gh_extractor_checkpointed.py --repos-file repos.txt \
      --out-base data/gh_comments/jan2023 --mode window \
      --since 2023-01-01T00:00:00Z --until 2023-02-01T00:00:00Z

  # Filtro por TÍTULO (usa defaults + añade propias):
  python gh_extractor_checkpointed.py --repos-file repos.txt \
      --out-base data/gh_comments/filtered --mode days --days 90 \
      --use-default-title-keywords \
      --title-keywords "panic,heap overflow"
"""

import os
import sys
import time
import argparse
import random
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Tuple
from http.client import RemoteDisconnected

import requests
import csv
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from urllib3.util import Retry
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(filename='.env', usecwd=True), override=True)

# ---------- Config ----------
GRAPHQL_ENDPOINT = "https://api.github.com/graphql"
PAGE_SIZE = 100  # (máximo permitido)

# Repos por defecto si no se pasa --repos-file
DEFAULT_REPOS = [
    "kubernetes/kubernetes",
    "kubernetes/ingress-nginx",
    "envoyproxy/envoy",
    "grafana/grafana",
    "prometheus/prometheus",
    "vercel/next.js",
    "nodejs/node",
    "tensorflow/tensorflow",
    "pytorch/pytorch",
    "openssl/openssl",
    "electron-userland/electron-builder",
]

# Palabras clave por defecto para títulos (ES/EN) — se activan con --use-default-title-keywords
DEFAULT_TITLE_KEYWORDS = [
    # genéricos bug/error/fallo
    "bug", "error", "failure", "fault", "issue", "regression", "crash", "hang", "freeze",
    "segfault", "panic", "unexpected", "incorrect", "wrong", "mismatch",
    "null pointer", "nil pointer", "undefined behavior", "UB", "race condition",
    "deadlock", "livelock", "timeout", "latency spike",
    "memory leak", "leak", "oom", "out of memory", "high memory", "use-after-free",
    "double free", "dangling", "buffer overflow", "stack overflow", "heap overflow",
    "out-of-bounds", "oob", "off-by-one", "overread", "overwrite", "corruption", "corrupt",
    "infinite loop", "loop", "recursion", "stack exhaustion",
    # seguridad
    "security", "vulnerability", "vuln", "cve", "exploit", "rce", "lpe", "privilege escalation",
    "xss", "cross-site scripting", "csrf", "ssrf", "sqli", "sql injection",
    "command injection", "code injection", "template injection", "path traversal",
    "directory traversal", "insecure", "exposure", "information disclosure",
    "auth bypass", "authentication bypass", "authorization bypass",
    "weak cryptography", "weak crypto", "insecure default", "hardcoded secret",
    "sensitive data", "token leak", "credential leak",
]

# ---------- Utilidades base ----------
def require_token() -> str:
    tok = os.getenv("GITHUB_TOKEN")
    if not tok:
        print("ERROR: define GITHUB_TOKEN en el entorno.")
        sys.exit(1)
    return tok

def mk_session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
        "User-Agent": "gh-comments-extractor-checkpointed"
    })

    retry = Retry(
        total=8, connect=8, read=8, status=8,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(['POST']),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=64, pool_block=True)
    s.mount("https://", adapter); s.mount("http://", adapter)
    return s

def gql(session: requests.Session, query: str, variables: Dict[str, Any], backoff: float = 1.0) -> Dict[str, Any]:
    while True:
        try:
            r = session.post(
                GRAPHQL_ENDPOINT,
                json={"query": query, "variables": variables},
                timeout=(5, 40)
            )
            # Caso 403 explícito (rate limit core)
            if r.status_code == 403 and "rate limit" in r.text.lower():
                reset = r.headers.get("X-RateLimit-Reset")
                if reset and str(reset).isdigit():
                    wait = max(5, int(reset) - int(time.time()) + 1)
                    print(f"[gql] 403 rate limit. Duermo {wait}s…")
                    time.sleep(wait)
                else:
                    print("[gql] 403 rate limit sin cabecera 'reset'; consulto rateLimit y duermo hasta reset…")
                    sleep_until_reset(session, fallback_seconds=60)
                continue

            if r.status_code == 200:
                data = r.json()
                if "errors" in data and data["errors"]:
                    msgs = " | ".join(e.get("message","") for e in data["errors"])
                    low = msgs.lower()
                    # Mensajes típicos de límite/abuso en errores GraphQL
                    if ("rate limit" in low) or ("secondary rate" in low) or ("abuse" in low):
                        print(f"[gql] Error de cuota/abuso: {msgs}. Duermo hasta reset…")
                        sleep_until_reset(session, fallback_seconds=60)
                        continue
                    # Error transitorio -> backoff exponencial corto
                    if "something went wrong" in msgs.lower():
                        sleep_for = min(backoff * 1.7 + random.uniform(0, 0.5), 30.0)
                        print(f"[gql] Error transitorio: {msgs}. Reintento en {sleep_for:.1f}s…")
                        time.sleep(sleep_for); backoff = sleep_for; continue
                    # Error duro
                    raise RuntimeError(f"GraphQL error: {msgs}")
                return data["data"]

            if r.status_code in (429, 500, 502, 503, 504):
                # 429 too many requests
                if r.status_code == 429:
                    print("[gql] 429 Too Many Requests. Duermo hasta reset…")
                    sleep_until_reset(session, fallback_seconds=60)
                    continue
                # Otras 5xx transitorias
                sleep_for = min(backoff * 1.7 + random.uniform(0, 0.5), 30.0)
                print(f"[gql] HTTP {r.status_code}. Reintento en {sleep_for:.1f}s…")
                time.sleep(sleep_for); backoff = sleep_for; continue

            # Otros códigos
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")

        except (RequestException, RemoteDisconnected) as e:
            sleep_for = min(backoff * 1.7 + random.uniform(0, 0.5), 30.0)
            print(f"[gql] Conexión abortada ({e}). Reintento en {sleep_for:.1f}s…")
            time.sleep(sleep_for); backoff = sleep_for; continue

def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def parse_iso(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()

def parse_iso_optional(s: Optional[str]) -> Optional[float]:
    if not s: return None
    try:
        return parse_iso(s)
    except Exception:
        raise ValueError(f"Fecha inválida, usa ISO-8601, p.ej. 2024-01-31T00:00:00Z: {s}")

# ---------- GraphQL ----------
ISSUE_BODY_Q = """
query IssueBody($owner:String!, $name:String!, $number:Int!) {
  repository(owner:$owner, name:$name) {
    issue(number:$number) {
      number title url state createdAt updatedAt
      author { login }
      labels(first:50){nodes{name}}
      bodyText
    }
  }
}
"""

PR_BODY_Q = """
query PRBody($owner:String!, $name:String!, $number:Int!) {
  repository(owner:$owner, name:$name) {
    pullRequest(number:$number) {
      number title url state createdAt updatedAt mergedAt
      author { login }
      labels(first:50){nodes{name}}
      bodyText
    }
  }
}
"""

ISSUES_LIST_Q = """
query Issues($owner:String!, $name:String!, $pageSize:Int!, $cursor:String) {
  repository(owner:$owner, name:$name) {
    issues(first:$pageSize, after:$cursor, orderBy:{field:UPDATED_AT, direction:DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title url state createdAt updatedAt
        author { login }
        labels(first:50){ nodes { name } }
        bodyText
      }
    }
  }
}
"""

PRS_LIST_Q = """
query PRs($owner:String!, $name:String!, $pageSize:Int!, $cursor:String) {
  repository(owner:$owner, name:$name) {
    pullRequests(first:$pageSize, after:$cursor, orderBy:{field:UPDATED_AT, direction:DESC}, states:[OPEN, MERGED, CLOSED]) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title url state createdAt updatedAt mergedAt
        author { login }
        labels(first:50){ nodes { name } }
        bodyText
      }
    }
  }
}
"""

ISSUE_COMMENTS_Q = """
query IssueComments($owner:String!, $name:String!, $number:Int!, $pageSize:Int!, $cursor:String) {
  repository(owner:$owner, name:$name) {
    issue(number:$number) {
      title url state createdAt updatedAt
      labels(first:50){nodes{name}}
      comments(first:$pageSize, after:$cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id url bodyText createdAt updatedAt isMinimized
          author { login }
        }
      }
    }
  }
}
"""

PR_COMMENTS_Q = """
query PRComments($owner:String!, $name:String!, $number:Int!, $pageSize:Int!, $cursor:String) {
  repository(owner:$owner, name:$name) {
    pullRequest(number:$number) {
      title url state createdAt updatedAt mergedAt
      labels(first:50){nodes{name}}
      comments(first:$pageSize, after:$cursor) {
        pageInfo { hasNextPage endCursor }
        nodes { id url bodyText createdAt updatedAt isMinimized author { login } }
      }
    }
  }
}
"""

PR_REVIEW_THREADS_Q = """
query PRReviewThreads($owner:String!, $name:String!, $number:Int!, $pageSize:Int!, $cursor:String) {
  repository(owner:$owner, name:$name) {
    pullRequest(number:$number) {
      reviewThreads(first:$pageSize, after:$cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id isResolved
          comments(first:100) {
            nodes { id url bodyText createdAt updatedAt author { login } }
          }
        }
      }
    }
  }
}
"""

# ---------- Iteradores ----------
def list_issues(session, owner, repo) -> Iterable[Dict[str, Any]]:
    cursor = None
    while True:
        data = gql(session, ISSUES_LIST_Q, {"owner":owner, "name":repo, "pageSize":PAGE_SIZE, "cursor":cursor})
        conn = data["repository"]["issues"]
        for n in conn["nodes"]:
            yield n
        if not conn["pageInfo"]["hasNextPage"]: break
        cursor = conn["pageInfo"]["endCursor"]
        time.sleep(0.4)

def list_prs(session, owner, repo) -> Iterable[Dict[str, Any]]:
    cursor = None
    while True:
        data = gql(session, PRS_LIST_Q, {"owner":owner, "name":repo, "pageSize":PAGE_SIZE, "cursor":cursor})
        conn = data["repository"]["pullRequests"]
        for n in conn["nodes"]:
            yield n
        if not conn["pageInfo"]["hasNextPage"]: break
        cursor = conn["pageInfo"]["endCursor"]
        time.sleep(0.4)

def iter_issue_comments(session, owner, repo, number) -> Iterable[Dict[str, Any]]:
    cursor = None
    while True:
        data = gql(session, ISSUE_COMMENTS_Q, {"owner":owner,"name":repo,"number":number,"pageSize":PAGE_SIZE,"cursor":cursor})
        issue = data["repository"]["issue"]
        if not issue: break
        for n in issue["comments"]["nodes"]:
            yield issue, n
        if not issue["comments"]["pageInfo"]["hasNextPage"]: break
        cursor = issue["comments"]["pageInfo"]["endCursor"]

def iter_pr_issue_comments(session, owner, repo, number) -> Iterable[Dict[str, Any]]:
    cursor = None
    while True:
        data = gql(session, PR_COMMENTS_Q, {"owner":owner,"name":repo,"number":number,"pageSize":PAGE_SIZE,"cursor":cursor})
        pr = data["repository"]["pullRequest"]
        if not pr: break
        for n in pr["comments"]["nodes"]:
            yield pr, n
        if not pr["comments"]["pageInfo"]["hasNextPage"]: break
        cursor = pr["comments"]["pageInfo"]["endCursor"]

def iter_pr_review_comments(session, owner, repo, number) -> Iterable[Dict[str, Any]]:
    cursor = None
    while True:
        data = gql(session, PR_REVIEW_THREADS_Q, {"owner":owner,"name":repo,"number":number,"pageSize":50,"cursor":cursor})
        pr = data["repository"]["pullRequest"]
        if not pr: break
        conn = pr["reviewThreads"]
        for thread in conn["nodes"]:
            tid = thread["id"]
            for c in thread["comments"]["nodes"]:
                yield tid, c
        if not conn["pageInfo"]["hasNextPage"]: break
        cursor = conn["pageInfo"]["endCursor"]

# ---------- Transformaciones ----------
def row_issue_body_from_node(repo_full: str, node: dict) -> dict:
    return {
        "id": f"github_issuebody_{repo_full}#{node['number']}",
        "platform": "github",
        "text": node.get("bodyText") or "",
        "created_at": node["createdAt"],
        "url": node["url"],
        "repo": repo_full,
        "issue_number": node["number"],
        "is_pr": False,
        "comment_type": "issue_body",
        "author": (node.get("author") or {}).get("login"),
        "context_id": f"{repo_full}#issue:{node['number']}",
        "container_title": node.get("title"),
        "container_state": node.get("state"),
        "container_url": node.get("url"),
        "container_created_at": node.get("createdAt"),
        "container_updated_at": node.get("updatedAt"),
        "container_labels": [lb["name"] for lb in (node.get("labels", {}) or {}).get("nodes", [])],
    }

def row_pr_body_from_node(repo_full: str, node: dict) -> dict:
    return {
        "id": f"github_prbody_{repo_full}#{node['number']}",
        "platform": "github",
        "text": node.get("bodyText") or "",
        "created_at": node["createdAt"],
        "url": node["url"],
        "repo": repo_full,
        "issue_number": node["number"],
        "is_pr": True,
        "comment_type": "pr_body",
        "author": (node.get("author") or {}).get("login"),
        "context_id": f"{repo_full}#pr:{node['number']}",
        "container_title": node.get("title"),
        "container_state": node.get("state"),
        "container_url": node.get("url"),
        "container_created_at": node.get("createdAt"),
        "container_updated_at": node.get("updatedAt"),
        "container_labels": [lb["name"] for lb in (node.get("labels", {}) or {}).get("nodes", [])],
    }

def row_issue(repo_full:str, number:int, issue:Dict[str,Any], node:Dict[str,Any]) -> Dict[str,Any]:
    return {
        "id": f"github_issuecomment_{node['id']}",
        "platform":"github",
        "text": node.get("bodyText") or "",
        "created_at": node["createdAt"],
        "url": node["url"],
        "repo": repo_full,
        "issue_number": number,
        "is_pr": False,
        "comment_type":"issue_comment",
        "author": (node.get("author") or {}).get("login"),
        "context_id": f"{repo_full}#issue:{number}",
        "container_title": issue.get("title"),
        "container_state": issue.get("state"),
        "container_url": issue.get("url"),
        "container_created_at": issue.get("createdAt"),
        "container_updated_at": issue.get("updatedAt"),
        "container_labels":[lb["name"] for lb in (issue.get("labels",{}) or {}).get("nodes",[])],
    }

def row_pr_issue(repo_full:str, number:int, pr:Dict[str,Any], node:Dict[str,Any]) -> Dict[str,Any]:
    return {
        "id": f"github_pr_issuecomment_{node['id']}",
        "platform":"github",
        "text": node.get("bodyText") or "",
        "created_at": node["createdAt"],
        "url": node["url"],
        "repo": repo_full,
        "issue_number": number,
        "is_pr": True,
        "comment_type":"issue_comment",
        "author": (node.get("author") or {}).get("login"),
        "context_id": f"{repo_full}#pr:{number}",
        "container_title": pr.get("title"),
        "container_state": pr.get("state"),
        "container_url": pr.get("url"),
        "container_created_at": pr.get("createdAt"),
        "container_updated_at": pr.get("updatedAt"),
        "container_labels":[lb["name"] for lb in (pr.get("labels",{}) or {}).get("nodes",[])],
    }

def row_pr_review(repo_full:str, number:int, node:Dict[str,Any], thread_id:str) -> Dict[str,Any]:
    return {
        "id": f"github_pr_reviewcomment_{node['id']}",
        "platform":"github",
        "text": node.get("bodyText") or "",
        "created_at": node["createdAt"],
        "url": node["url"],
        "repo": repo_full,
        "issue_number": number,
        "is_pr": True,
        "comment_type":"review_comment",
        "author": (node.get("author") or {}).get("login"),
        "context_id": f"{repo_full}#pr:{number}",
        "thread_id": thread_id,
    }

# ---------- Writer con hook de estado ----------
class Writer:
    def __init__(self, base: Path, dedupe: bool = True, on_write=None):
        self.csv_path = base.with_suffix(".csv")
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_new = not self.csv_path.exists()
        self._csv = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._w = csv.writer(self._csv)
        self._on_write = on_write  # <- hook opcional

        if self._csv_new:
            self._w.writerow([
                "repo","is_pr","issue_number","comment_type","comment_id","comment_created_at","comment_author",
                "text","comment_url","context_id","container_title","container_state","container_url","container_created_at","container_updated_at","container_labels"
            ])

        # dedupe previa
        self.seen_ids = set()
        if dedupe and self.csv_path.exists():
            try:
                with open(self.csv_path, newline="", encoding="utf-8") as f:
                    r = csv.reader(f)
                    header = next(r, None)
                    idx = header.index("comment_id") if header and "comment_id" in header else 4
                    for row in r:
                        if row:
                            self.seen_ids.add(row[idx])
            except Exception as e:
                print(f"[dedupe] No se pudo precargar IDs: {e}")

    def write_row(self, row: Dict[str, Any]):
        self._w.writerow([
            row["repo"], row["is_pr"], row["issue_number"], row["comment_type"], row["id"], row["created_at"], row["author"],
            row["text"], row["url"], row["context_id"], row["container_title"], row["container_state"], row["container_url"], row["container_created_at"], row["container_updated_at"], ";".join(row.get("container_labels",[]))
        ])
        if self._on_write:
            try:
                self._on_write(row)
            except Exception:
                pass

    def write_unique(self, row: Dict[str, Any]) -> bool:
        rid = row["id"]
        if rid in self.seen_ids:
            return False
        self.write_row(row)
        self.seen_ids.add(rid)
        return True

    def flush(self):
        self._csv.flush()

    def close(self):
        self._csv.close()

# ---------- Estado (state.json) ----------
def load_state(path: Path) -> Dict[str, Dict[str, float]]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def save_state(path: Path, state: Dict[str, Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

# ---------- Ventanas ----------
def compute_window(mode: str,
                   days: Optional[int],
                   since: Optional[str],
                   until: Optional[str],
                   from_days: Optional[int],
                   to_days: Optional[int],
                   repo_full: str,
                   state: Dict[str, Dict[str, float]]) -> Tuple[float, float]:
    """
    Devuelve (low_ts, high_ts) INCLUSIVOS.
    - days   : [now - days, now]
    - newer  : [newest_seen+ε, until or now] (si no hay state, cae a days o 365d)
    - older  : [hi - days, hi] con hi = oldest_seen-ε  (si no hay state, cae a [now-days, now])
    - range  : [now - from_days, now - to_days]
    - window : [since, until] con ISO (faltantes caen a [now-365d, now] o al que haya)
    """
    now_ts = datetime.now(timezone.utc).timestamp()
    eps = 1.0

    s_since = parse_iso_optional(since)
    s_until = parse_iso_optional(until)

    rep = state.get(repo_full, {})
    newest = rep.get("newest_comment_ts")
    oldest = rep.get("oldest_comment_ts")

    if mode == "days":
        d = days if days is not None else 365
        return now_ts - d*86400, now_ts

    if mode == "newer":
        lo = (newest + eps) if (newest is not None) else (now_ts - (days or 365)*86400)
        hi = s_until if s_until is not None else now_ts
        return lo, hi

    if mode == "older":
        hi = (oldest - eps) if (oldest is not None) else now_ts
        d = days if days is not None else 365
        lo = hi - d*86400
        if lo > hi: lo = hi - 1
        return lo, hi

    if mode == "range":
        if from_days is None or to_days is None:
            raise ValueError("--mode range requiere --from-days y --to-days")
        if from_days < to_days:
            raise ValueError("--from-days debe ser >= --to-days (p.ej. 30 y 7)")
        lo = now_ts - from_days*86400
        hi = now_ts - to_days*86400
        if lo > hi: lo, hi = hi, lo
        return lo, hi

    if mode == "window":
        hi = s_until if s_until is not None else now_ts
        lo = s_since if s_since is not None else (hi - (days or 365)*86400)
        if lo > hi: lo, hi = hi - 1, hi
        return lo, hi

    raise ValueError(f"Modo no soportado: {mode}")

# ---------- Helpers de filtro por título ----------
def _normalize_keywords(seq: Iterable[str]) -> List[str]:
    out = []
    for s in seq:
        ss = s.strip().lower()
        if ss:
            out.append(ss)
    return out

def title_matches(title: Optional[str], patterns: List[str]) -> bool:
    if not patterns:
        return True  # sin filtro -> pasa todo
    t = (title or "").lower()
    return any(p in t for p in patterns)

# ---------- Núcleo extractor ----------
def extractor_repo(session,
                   owner: str,
                   repo: str,
                   writer: Writer,
                   low_ts: float,
                   high_ts: float,
                   mode: str,
                   include_reviews: bool,
                   only_initial_post: bool,
                   max_comments_per_item: Optional[int],
                   max_comments_per_repo: Optional[int],
                   state: Dict[str, Dict[str, float]],
                   title_keywords: List[str]) -> int:
    """
    Extrae comentarios cuyo createdAt cae en [low_ts, high_ts].
    Aplica filtro por TÍTULO de issue/PR (si hay keywords).
    Actualiza state[repo]['newest_comment_ts'/'oldest_comment_ts'] cuando escribe filas.
    Devuelve nº de comentarios NUEVOS escritos (tras dedupe).
    """
    repo_full = f"{owner}/{repo}"
    wrote_count = 0

    # Hook para actualizar min/max por repo
    def _on_write(row: Dict[str, Any]):
        created_ts = parse_iso(row["created_at"])
        entry = state.setdefault(repo_full, {})
        if "newest_comment_ts" not in entry or created_ts > entry["newest_comment_ts"]:
            entry["newest_comment_ts"] = created_ts
        if "oldest_comment_ts" not in entry or created_ts < entry["oldest_comment_ts"]:
            entry["oldest_comment_ts"] = created_ts

    if not hasattr(writer, "_on_write") or writer._on_write is None:
        writer._on_write = _on_write

    # Normalización
    if max_comments_per_item is not None and max_comments_per_item < 0:
        max_comments_per_item = 0

    repo_budget = max_comments_per_repo
    def can_write_more() -> bool:
        return (repo_budget is None) or (repo_budget > 0)

    def in_window(ts_iso: str) -> bool:
        ts = parse_iso(ts_iso)
        return (low_ts <= ts <= high_ts)

    # -------- Issues --------
    for it in list_issues(session, owner, repo):
        # corte rápido en "newer"
        if mode == "newer" and parse_iso(it["updatedAt"]) < low_ts:
            break

        # FILTRO por título (issue)
        if not title_matches(it.get("title"), title_keywords):
            continue

        if only_initial_post:
            if in_window(it["createdAt"]):
                if writer.write_unique(row_issue_body_from_node(repo_full, it)):
                    wrote_count += 1
            continue

        if not can_write_more(): break
        if max_comments_per_item == 0: continue

        count = 0
        num = it["number"]
        for issue, com in iter_issue_comments(session, owner, repo, num):
            if in_window(com["createdAt"]):
                if not can_write_more(): break
                if (max_comments_per_item is not None) and (count >= max_comments_per_item): break
                row = row_issue(repo_full, num, issue, com)
                if writer.write_unique(row):
                    count += 1
                    wrote_count += 1
                    if repo_budget is not None: repo_budget -= 1
                if not can_write_more(): break

        if not can_write_more(): break

    # -------- PRs --------
    for prn in list_prs(session, owner, repo):
        if mode == "newer" and parse_iso(prn["updatedAt"]) < low_ts:
            break

        # FILTRO por título (PR)
        if not title_matches(prn.get("title"), title_keywords):
            continue

        if only_initial_post:
            if in_window(prn["createdAt"]):
                if writer.write_unique(row_pr_body_from_node(repo_full, prn)):
                    wrote_count += 1
            continue

        if not can_write_more(): break
        if max_comments_per_item == 0: continue

        count = 0
        num = prn["number"]

        # PR issue comments (Conversation)
        for pr, com in iter_pr_issue_comments(session, owner, repo, num):
            if in_window(com["createdAt"]):
                if not can_write_more(): break
                if (max_comments_per_item is not None) and (count >= max_comments_per_item): break
                row = row_pr_issue(repo_full, num, pr, com)
                if writer.write_unique(row):
                    count += 1
                    wrote_count += 1
                    if repo_budget is not None: repo_budget -= 1
                if not can_write_more(): break
        if not can_write_more(): break

        # Review comments
        if include_reviews and (max_comments_per_item is None or count < max_comments_per_item):
            for thread_id, rc in iter_pr_review_comments(session, owner, repo, num):
                if in_window(rc["CreatedAt"] if "CreatedAt" in rc else rc["createdAt"]):
                    if not can_write_more(): break
                    if (max_comments_per_item is not None) and (count >= max_comments_per_item): break
                    row = row_pr_review(repo_full, num, rc, thread_id)
                    if writer.write_unique(row):
                        count += 1
                        wrote_count += 1
                        if repo_budget is not None: repo_budget -= 1
                    if not can_write_more(): break
        if not can_write_more(): break

    return wrote_count

# ---------- Repos file ----------
def read_repos_file(p: Optional[str]) -> List[str]:
    if not p: return DEFAULT_REPOS
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos-file", type=str, help="Archivo con repos owner/name por línea.")
    ap.add_argument("--out-base", type=str, default="../../data/gh_comments/training/gh_comments_2023_now",
                    help="Ruta base sin extensión para el CSV y el state (.csv y .state.json).")

    # Modos de ventana
    ap.add_argument("--mode", type=str, choices=["days", "newer", "older", "range", "window"], default="days",
                    help="days: hoy→N días atrás | newer: desde lo último visto→hoy | older: N días antes de lo más viejo visto→lo más viejo | range: [from-days, to-days] | window: [since, until] ISO")
    ap.add_argument("--days", type=int, default=None, help="N días (para 'days' y 'older'; fallback en 'newer/window' si faltan fechas).")
    ap.add_argument("--from-days", type=int, default=None, help="En 'range': desde 'from-days' atrás (ej. 30).")
    ap.add_argument("--to-days", type=int, default=None, help="En 'range': hasta 'to-days' atrás (ej. 7).")
    ap.add_argument("--since", type=str, default=None, help="En 'window': ISO-8601 inferior (ej. 2023-01-01T00:00:00Z)")
    ap.add_argument("--until", type=str, default=None, help="En 'window'/'newer': ISO-8601 superior (por defecto ahora)")

    # Qué comentarios incluir
    ap.add_argument("--only-initial-post", action="store_true", help="Guardar sólo el body inicial de Issues/PRs (sin comentarios).")
    ap.add_argument("--also-include-initial-post", action="store_true", help="Además de los comentarios, guarda también el body de issues/PRs.")
    ap.add_argument("--include-review-comments", action="store_true", help="Incluir comentarios de code review en PRs.")
    ap.add_argument("--max-comments-per-item", type=int, default=None, help="Máximo de comentarios por Issue/PR (0 = ninguno).")
    ap.add_argument("--max-comments-per-repo", type=int, default=None, help="Límite total de comentarios a extraer por repo.")

    # Filtro por título
    ap.add_argument("--use-default-title-keywords", action="store_true",
                    help="Activa un conjunto por defecto de palabras clave (bug/vuln ES/EN) para filtrar por título.")
    ap.add_argument("--title-keywords", type=str, default=None,
                    help="Lista coma-separada de palabras/frases a buscar en el título (case-insensitive).")
    ap.add_argument("--title-keywords-file", type=str, default=None,
                    help="Ruta de archivo con una palabra/frase por línea para filtrar por título.")

    # State
    ap.add_argument("--state-path", type=str, default=None, help="Ruta del .state.json (por defecto <out-base>.state.json)")

    args = ap.parse_args()

    # Preparar keywords activas
    active_kw: List[str] = []
    if args.use_default_title_keywords:
        active_kw.extend(DEFAULT_TITLE_KEYWORDS)
    if args.title_keywords:
        active_kw.extend([x.strip() for x in args.title_keywords.split(",")])
    if args.title_keywords_file:
        fp = Path(args.title_keywords_file)
        if fp.exists():
            for line in fp.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    active_kw.append(line.strip())
        else:
            print(f"[warn] No existe --title-keywords-file: {fp}")
    active_kw = _normalize_keywords(active_kw)
    if active_kw:
        print(f"[info] Filtro activo por título con {len(active_kw)} palabras/frases.")

    repos = read_repos_file(args.repos_file)
    token = require_token()
    session = mk_session(token)

    base = Path(args.out_base)
    writer = Writer(base)
    state_path = Path(args.state_path) if args.state_path else base.with_suffix(".state.json")
    state = load_state(state_path)

    total = 0
    for repo_full in repos:
        try:
            owner, repo = repo_full.split("/", 1)
        except ValueError:
            print(f"Repo inválido: {repo_full}")
            continue

        low_ts, high_ts = compute_window(
            mode=args.mode,
            days=args.days,
            since=args.since,
            until=args.until,
            from_days=args.from_days,
            to_days=args.to_days,
            repo_full=repo_full,
            state=state
        )

        print(f"\n==> {repo_full} | modo={args.mode} | ventana [{iso(low_ts)} .. {iso(high_ts)}]")

        try:
            n_repo = extractor_repo(
                session=session,
                owner=owner, repo=repo,
                writer=writer,
                low_ts=low_ts, high_ts=high_ts,
                mode=args.mode,
                include_reviews=args.include_review_comments,
                only_initial_post=args.only_initial_post,
                max_comments_per_item=args.max_comments_per_item,
                max_comments_per_repo=args.max_comments_per_repo,
                state=state,
                title_keywords=active_kw
            )
            total += n_repo
            writer.flush()
        except Exception as e:
            print(f"  ! Error en {repo_full}: {e}")
            continue

    writer.close()
    save_state(state_path, state)
    print(f"\nExtracción completada. Comentarios nuevos escritos: {total}")
    print(f"CSV:   {base.with_suffix('.csv')}")
    print(f"STATE: {state_path}")

if __name__ == "__main__":
    main()
