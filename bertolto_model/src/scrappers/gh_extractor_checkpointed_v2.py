#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extractor de comentarios de GitHub con checkpoints y ventanas flexibles + filtro por título.
- Guarda un CSV incremental y un .state.json con el comentario más nuevo y más viejo vistos por repo.
- Filtra SIEMPRE por ventana temporal (createdAt ∈ [low_ts, high_ts]).
- Opcional: filtra por palabras clave en el TÍTULO de issues/PRs (container_title).
- NUEVO: flags para incluir body y seleccionar qué comentario por issue/PR (first/newest/most-reacted).

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
      --out-base data/gh_comments/lastyear --mode days --days 365

  # Incremental hacia ADELANTE desde lo último visto:
  python gh_extractor_checkpointed.py --repos-file repos.txt \
      --out-base data/gh_comments/lastyear --mode newer

  # Body + primer comentario (por fecha más antigua en ventana) de cada issue/PR que cumpla el filtro de título:
  python gh_extractor_checkpointed.py --repos-file repos.txt \
      --out-base data/gh_comments/filtered --mode days --days 90 \
      --use-default-title-keywords \
      --also-include-initial-post --max-comments-per-item 1 \
      --comment-selection first

  # Body + comentario con más reacciones:
  python gh_extractor_checkpointed.py --repos-file repos.txt \
      --out-base data/gh_comments/filtered --mode days --days 90 \
      --also-include-initial-post --max-comments-per-item 1 \
      --comment-selection most-reacted
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
    "envoyproxy/envoy",
    "grafana/grafana",
    "prometheus/prometheus",
    "vercel/next.js",
    "nodejs/node",
    "tensorflow/tensorflow",
    "pytorch/pytorch",
    "openssl/openssl",
    "electron-userland/electron-builder",
    "kubernetes/kubernetes",
    "kubernetes/ingress-nginx"
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
            # Throttle básico por petición
            if hasattr(session, "throttler") and session.throttler:
                session.throttler.wait()

            r = session.post(
                GRAPHQL_ENDPOINT,
                json={"query": query, "variables": variables},
                timeout=(5, 40)
            )

            # Revisa cabeceras core REST (GitHub también las manda en GraphQL)
            rem = r.headers.get("X-RateLimit-Remaining")
            rst = r.headers.get("X-RateLimit-Reset")
            if rem is not None:
                try:
                    rem_i = int(rem)
                    if rem_i <= 0:
                        if rst and str(rst).isdigit():
                            wait = max(5, int(rst) - int(time.time()) + 1)
                            print(f"[gql] X-RateLimit-Remaining=0. Duermo {wait}s hasta reset…")
                            time.sleep(wait)
                            continue
                except Exception:
                    pass

            # 403 explícito
            if r.status_code == 403 and "rate limit" in r.text.lower():
                print("[gql] 403 rate limit. Duermo hasta reset…")
                if hasattr(session, "throttler") and session.throttler:
                    session.throttler.penalize()
                sleep_until_reset(session, fallback_seconds=60)
                continue

            if r.status_code == 200:
                data = r.json()
                if "errors" in data and data["errors"]:
                    msgs = " | ".join(e.get("message","") for e in data["errors"])
                    low = msgs.lower()

                    # Señales típicas
                    if ("secondary rate" in low) or ("abuse" in low) or ("rate limit" in low):
                        print(f"[gql] Señal de cuota/abuso: {msgs}. Duermo hasta reset y penalizo throttle…")
                        if hasattr(session, "throttler") and session.throttler:
                            session.throttler.penalize()
                        sleep_until_reset(session, fallback_seconds=60)
                        continue

                    # error transitorio
                    if "something went wrong" in low:
                        sleep_for = min(backoff * 1.7 + random.uniform(0, 0.5), 30.0)
                        print(f"[gql] Error transitorio: {msgs}. Reintento en {sleep_for:.1f}s…")
                        time.sleep(sleep_for); backoff = sleep_for; continue

                    # error duro
                    raise RuntimeError(f"GraphQL error: {msgs}")

                # OK
                return data["data"]

            if r.status_code in (429, 500, 502, 503, 504):
                if r.status_code == 429:
                    print("[gql] 429 Too Many Requests. Duermo hasta reset…")
                    if hasattr(session, "throttler") and session.throttler:
                        session.throttler.penalize()
                    sleep_until_reset(session, fallback_seconds=60)
                    continue
                # otras 5xx
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

# ---------- Rate limit helpers & throttling ----------

RATE_LIMIT_Q = """
query { rateLimit { limit cost remaining resetAt used } }
"""

def _post_raw(session: requests.Session, payload: dict, timeout=(5, 40)) -> requests.Response:
    # post directo sin recursión a gql()
    return session.post(GRAPHQL_ENDPOINT, json=payload, timeout=timeout)

def get_rate_limit(session: requests.Session) -> Optional[dict]:
    try:
        r = _post_raw(session, {"query": RATE_LIMIT_Q, "variables": {}}, timeout=(5, 20))
        if r.status_code == 200:
            j = r.json()
            return ((j or {}).get("data") or {}).get("rateLimit")
    except Exception:
        pass
    return None

def parse_iso_z(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()

def sleep_until_reset(session: requests.Session, fallback_seconds: float = 60.0):
    rl = get_rate_limit(session)
    if rl:
        remaining = rl.get("remaining")
        resetAt = rl.get("resetAt")
        if resetAt:
            now = time.time()
            tgt = parse_iso_z(resetAt)
            wait = max(5.0, tgt - now + 1.0)
            print(f"[rate-limit] remaining={remaining}, resetAt={resetAt}. Duermo {wait:.1f}s…")
            time.sleep(wait)
            return
    # si no hay datos fiables:
    print(f"[rate-limit] No se pudo leer rateLimit. Duermo {fallback_seconds:.1f}s…")
    time.sleep(fallback_seconds)

class Throttler:
    """Evita golpear la API muy rápido y reacciona al 'secondary rate limit'."""
    def __init__(self, min_interval_sec: float = 0.30, max_interval_sec: float = 2.0, factor: float = 1.5):
        self.min_interval = float(min_interval_sec)
        self.max_interval = float(max_interval_sec)
        self.factor = float(factor)
        self._last = 0.0

    def wait(self):
        now = time.time()
        delta = self.min_interval - (now - self._last)
        if delta > 0:
            time.sleep(delta)
        self._last = time.time()

    def penalize(self):
        # subir intervalo tras señales de abuso/sec rate limit
        self.min_interval = min(self.max_interval, self.min_interval * self.factor)
        print(f"[throttle] Intervalo elevado a {self.min_interval:.2f}s")

    def relax(self):
        # opcional: podrías relajarlo si todo va bien mucho tiempo
        pass


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

# Añadimos reactionGroups para poder calcular "most-reacted"
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
          reactionGroups { content users { totalCount } }
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
        nodes {
          id url bodyText createdAt updatedAt isMinimized
          author { login }
          reactionGroups { content users { totalCount } }
        }
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
            nodes {
              id url bodyText createdAt updatedAt
              author { login }
              reactionGroups { content users { totalCount } }
            }
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
        time.sleep(0.6)

def list_prs(session, owner, repo) -> Iterable[Dict[str, Any]]:
    cursor = None
    while True:
        data = gql(session, PRS_LIST_Q, {"owner":owner, "name":repo, "pageSize":PAGE_SIZE, "cursor":cursor})
        conn = data["repository"]["pullRequests"]
        for n in conn["nodes"]:
            yield n
        if not conn["pageInfo"]["hasNextPage"]: break
        cursor = conn["pageInfo"]["endCursor"]
        time.sleep(0.6)

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
def _sum_reactions(node: Dict[str, Any]) -> int:
    groups = node.get("reactionGroups") or []
    total = 0
    for g in groups:
        users = (g.get("users") or {}).get("totalCount", 0)
        try:
            total += int(users)
        except Exception:
            pass
    return total

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

# ---------- Selección de comentarios por ítem ----------
def _select_comments(cands: List[Tuple[Dict[str,Any], float, int]],
                     policy: str,
                     k: int) -> List[Dict[str,Any]]:
    """
    cands: lista de (row, created_ts, reactions_sum)
    policy: 'first'/'oldest' (creación asc), 'newest' (desc), 'most-reacted' (reacciones desc, tie-break fecha desc)
    """
    if not cands or k <= 0:
        return []
    if policy in ("first", "oldest"):
        cands.sort(key=lambda x: x[1])                      # más antiguo primero
    elif policy == "newest":
        cands.sort(key=lambda x: x[1], reverse=True)        # más nuevo primero
    elif policy == "most-reacted":
        cands.sort(key=lambda x: (x[2], x[1]), reverse=True)  # más reacciones, y si empate el más nuevo
    else:  # fallback sensato
        cands.sort(key=lambda x: x[1])
    return [r for (r, _, _) in cands[:k]]

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
                   also_include_initial_post: bool,
                   max_comments_per_item: Optional[int],
                   max_comments_per_repo: Optional[int],
                   state: Dict[str, Dict[str, float]],
                   title_keywords: List[str],
                   comment_selection: str) -> int:
    """
    Extrae comentarios cuyo createdAt cae en [low_ts, high_ts].
    Aplica filtro por TÍTULO de issue/PR (si hay keywords).
    Permite incluir body además de comentarios y seleccionar el comentario por ítem.
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

        # Body (además de comentarios) si está activado
        if also_include_initial_post and in_window(it["createdAt"]):
            if writer.write_unique(row_issue_body_from_node(repo_full, it)):
                wrote_count += 1

        if only_initial_post:
            # Si sólo se quieren bodies, ya seguimos al siguiente ítem
            continue

        if not can_write_more():
            break
        if max_comments_per_item == 0:
            continue

        num = it["number"]
        # Coleccionamos candidatos dentro de la ventana y luego seleccionamos según policy
        cands: List[Tuple[Dict[str,Any], float, int]] = []
        for issue, com in iter_issue_comments(session, owner, repo, num):
            if in_window(com["createdAt"]):
                row = row_issue(repo_full, num, issue, com)
                created_ts = parse_iso(com["createdAt"])
                reacts = _sum_reactions(com)
                cands.append((row, created_ts, reacts))

        # Seleccionar y escribir
        k = max_comments_per_item if max_comments_per_item is not None else len(cands)
        sel = _select_comments(cands, comment_selection, k)
        for row in sel:
            if not can_write_more(): break
            if writer.write_unique(row):
                wrote_count += 1
                if repo_budget is not None: repo_budget -= 1
        if not can_write_more(): break

    # -------- PRs --------
    for prn in list_prs(session, owner, repo):
        if mode == "newer" and parse_iso(prn["updatedAt"]) < low_ts:
            break

        # FILTRO por título (PR)
        if not title_matches(prn.get("title"), title_keywords):
            continue

        if also_include_initial_post and in_window(prn["createdAt"]):
            if writer.write_unique(row_pr_body_from_node(repo_full, prn)):
                wrote_count += 1

        if only_initial_post:
            continue

        if not can_write_more():
            break
        if max_comments_per_item == 0:
            continue

        num = prn["number"]

        # PR issue comments (Conversation)
        cands: List[Tuple[Dict[str,Any], float, int]] = []
        for pr, com in iter_pr_issue_comments(session, owner, repo, num):
            if in_window(com["createdAt"]):
                row = row_pr_issue(repo_full, num, pr, com)
                created_ts = parse_iso(com["createdAt"])
                reacts = _sum_reactions(com)
                cands.append((row, created_ts, reacts))

        # Review comments (opcionales)
        if include_reviews:
            for thread_id, rc in iter_pr_review_comments(session, owner, repo, num):
                ts_key = "createdAt" if "createdAt" in rc else "CreatedAt"
                if in_window(rc[ts_key]):
                    row = row_pr_review(repo_full, num, rc, thread_id)
                    created_ts = parse_iso(rc[ts_key])
                    reacts = _sum_reactions(rc)
                    cands.append((row, created_ts, reacts))

        k = max_comments_per_item if max_comments_per_item is not None else len(cands)
        sel = _select_comments(cands, comment_selection, k)
        for row in sel:
            if not can_write_more(): break
            if writer.write_unique(row):
                wrote_count += 1
                if repo_budget is not None: repo_budget -= 1
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
    ap.add_argument("--rq-interval-ms", type=int, default=350,
                    help="Separación mínima entre peticiones GraphQL (ms). Previene secondary rate limit. Default 350ms.")

    # Modos de ventana
    # Primero hago una pasada desde 2023 con max-comments-per-repo de 3000
    # python gh_extractor_checkpointed_v2.py --mode window --since 2023-01-01T00:00:00Z --also-include-initial-post --max-comments-per-item 1 --comment-selection most-reacted --use-default-title-keywords

    # Segunda pasada sin límite utilizando --newer para obtener el resto de comentarios de cada repo desde 2023
    # Si vuelve a petar, hacer varias pasadas desde 2023 con limite de comentarios por repo
    ap.add_argument("--mode", type=str, choices=["days", "newer", "older", "range", "window"], default="days",
                    help="days: hoy→N días atrás | newer: desde lo último visto→hoy | older: N días antes de lo más viejo visto→lo más viejo | range: [from-days, to-days] | window: [since, until] ISO")
    ap.add_argument("--days", type=int, default=None, help="N días (para 'days' y 'older'; fallback en 'newer/window' si faltan fechas).")
    ap.add_argument("--from-days", type=int, default=None, help="En 'range': desde 'from-days' atrás (ej. 30).")
    ap.add_argument("--to-days", type=int, default=None, help="En 'range': hasta 'to-days' atrás (ej. 7).")
    ap.add_argument("--since", type=str, default=None, help="En 'window': ISO-8601 inferior (ej. 2023-01-01T00:00:00Z)")
    ap.add_argument("--until", type=str, default=None, help="En 'window'/'newer': ISO-8601 superior (por defecto ahora)")

    # Qué incluir
    ap.add_argument("--only-initial-post", action="store_true",
                    help="Guardar sólo el body inicial de Issues/PRs (sin comentarios).")
    ap.add_argument("--also-include-initial-post", action="store_true",
                    help="Además de los comentarios, guarda también el body de issues/PRs.")
    ap.add_argument("--include-review-comments", action="store_true",
                    help="Incluir comentarios de code review en PRs.")
    ap.add_argument("--max-comments-per-item", type=int, default=None,
                    help="Máximo de comentarios por Issue/PR (0 = ninguno).")
    ap.add_argument("--max-comments-per-repo", type=int, default=None,
                    help="Límite total de comentarios a extraer por repo.")

    # Selección de comentarios por ítem
    ap.add_argument("--comment-selection", type=str,
                    choices=["first", "oldest", "newest", "most-reacted"],
                    default="first",
                    help="Cómo elegir comentarios por issue/PR cuando hay límite: "
                         "first/oldest (más antiguo en ventana), newest (más reciente), most-reacted (más reacciones).")

    # Filtro por título
    ap.add_argument("--use-default-title-keywords", action="store_true",
                    help="Activa un conjunto por defecto de palabras clave (bug/vuln ES/EN) para filtrar por título.")
    ap.add_argument("--title-keywords", type=str, default=None,
                    help="Lista coma-separada de palabras/frases a buscar en el título (case-insensitive).")
    ap.add_argument("--title-keywords-file", type=str, default=None,
                    help="Ruta de archivo con una palabra/frase por línea para filtrar por título.")

    # State
    ap.add_argument("--state-path", type=str, default=None,
                    help="Ruta del .state.json (por defecto <out-base>.state.json)")

    args = ap.parse_args()

    # Preparar keywords activas
    active_kw: List[str] = []
    if args.use_default_title_keyWORDS if False else args.use_default_title_keywords:
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

    # engancha throttler a la sesión
    try:
        session.throttler = Throttler(min_interval_sec=max(0.05, (args.rq_interval_ms or 350) / 1000.0)) # si sigue saltando "rate limit" subir de 350 a 500-800
    except Exception:
        session.throttler = Throttler(min_interval_sec=0.35)

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
                also_include_initial_post=args.also_include_initial_post,
                max_comments_per_item=args.max_comments_per_item,
                max_comments_per_repo=args.max_comments_per_repo,
                state=state,
                title_keywords=active_kw,
                comment_selection=args.comment_selection
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
