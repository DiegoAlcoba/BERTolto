#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extractor de comentarios de GitHub (últimos N días) para múltiples repos.
- Extrae issue comments (y opcionalmente review comments) vía GraphQL.
- Filtra por fecha del comentario (createdAt >= now - N días).
- Escribe CSV de forma incremental.

Uso:
  export GITHUB_TOKEN=ghp_xxx
  python gh_extractor_last_year.py --repos-file ../../../data/repos.txt --days 365 --include-review-comments

  # Solo el cuerpo de issues/PRs del último año
        python gh_extractor_last_year.py --repos-file ../../../data/repos.txt --days 365 --only-initial-post

    # O bien: máximo 1 comentario por issue/PR (además del body si luego lo activas)
        python gh_extractor_last_year.py --repos-file ../../../data/repos.txt --days 365 --max-comments-per-item N

"""

import os
import sys
import time
import argparse
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Tuple
from dotenv import load_dotenv, find_dotenv
from http.client import RemoteDisconnected
from requests.adapters import HTTPAdapter
from requests.exceptions import RequestException
from transformers.utils import can_return_tuple
from urllib3.util import Retry

# Carga el .env buscando desde el working directory hacia arriba
load_dotenv(find_dotenv(usecwd=True), override=True)

import requests
import csv

GRAPHQL_ENDPOINT = "https://api.github.com/graphql"
PAGE_SIZE = 100  # max 100

# Repos por defecto (puedes usar --repos-file para sobreescribir)
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

# Función que carga el token de GitHub
def require_token() -> str:
    tok = os.getenv("GITHUB_TOKEN")
    if not tok:
        print("ERROR: define GITHUB_TOKEN en el entorno.")
        sys.exit(1)
    return tok

# Función que crea la sesión en la API de GitHub con la librería requests
def mk_session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
        "User-Agent": "gh-comments-extractor-last-year"
    })

    retry = Retry (
        total=8, connect=8, read=8, status=8,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(['POST']),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_maxsize=64, pool_block=True)
    s.mount("https://", adapter); s.mount("http://", adapter)

    return s

# Función que lanza la petición contra la API de GitHub
def gql(session: requests.Session, query: str, variables: Dict[str, Any], backoff: float = 1.0) -> Dict[str, Any]:
    while True:
        try:
            r = session.post(
                GRAPHQL_ENDPOINT,
                json={"query": query, "variables": variables},
                timeout=(5, 40)  # 5s connect, 40s read
            )
            if r.status_code == 403 and "rate limit" in r.text.lower():
                reset = r.headers.get("X-RateLimit-Reset")
                sleep_for = max(5, int(reset) - int(time.time()) + 3) if (reset and reset.isdigit()) else 30
                print(f"[gql] 403 rate limit. Duermo {sleep_for}s…")
                time.sleep(sleep_for); continue

            if r.status_code == 200:
                data = r.json()
                if "errors" in data:
                    msgs = " | ".join(e.get("message","") for e in data["errors"])
                    if ("Something went wrong" in msgs) or ("abuse" in msgs.lower()) or ("rate limit" in msgs.lower()):
                        sleep_for = min(backoff * 1.7 + random.uniform(0, 0.5), 30.0)
                        print(f"[gql] Error transitorio: {msgs}. Reintento en {sleep_for:.1f}s…")
                        time.sleep(sleep_for); backoff = sleep_for; continue
                    raise RuntimeError(f"GraphQL error: {msgs}")
                return data["data"]

            if r.status_code in (429, 500, 502, 503, 504):
                sleep_for = min(backoff * 1.7 + random.uniform(0, 0.5), 30.0)
                print(f"[gql] HTTP {r.status_code}. Reintento en {sleep_for:.1f}s…")
                time.sleep(sleep_for); backoff = sleep_for; continue

            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")

        except (RequestException, RemoteDisconnected) as e:
            sleep_for = min(backoff * 1.7 + random.uniform(0, 0.5), 30.0)
            print(f"[gql] Conexión abortada ({e}). Reintento en {sleep_for:.1f}s…")
            time.sleep(sleep_for); backoff = sleep_for; continue

def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def parse_iso(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()

# Modelos de querys GraphQL
# Query para obtener el body (comentario ppal) de un Issue

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

# Query para obtener el body (comentario ppal) de un PR
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


# Lista de Issues de un repositorio
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

# Lista de PRs en un repositorio
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

# Pide los comentarios de un Issue (number) para recuperar comentarios puros
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

# Pide comentarios de conversación de un PR (number)
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

# Pide los hilos de code review de un PR (number) y todos sus comentarios
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

# ITERADORES -> Cómo se consumen las queries

# Función que pagina la lista de Issues
def list_issues(session, owner, repo) -> Iterable[Dict[str, Any]]:
    cursor = None
    while True:
        data = gql(session, ISSUES_LIST_Q, {"owner":owner, "name":repo, "pageSize":PAGE_SIZE, "cursor":cursor})
        conn = data["repository"]["issues"]
        for n in conn["nodes"]:
            yield n
        if not conn["pageInfo"]["hasNextPage"]: break
        cursor = conn["pageInfo"]["endCursor"]
        time.sleep(0.2) #Para evitar timeouts

# Función que pagina la lista de PRs
def list_prs(session, owner, repo) -> Iterable[Dict[str, Any]]:
    cursor = None
    while True:
        data = gql(session, PRS_LIST_Q, {"owner":owner, "name":repo, "pageSize":PAGE_SIZE, "cursor":cursor})
        conn = data["repository"]["pullRequests"]
        for n in conn["nodes"]:
            yield n
        if not conn["pageInfo"]["hasNextPage"]: break
        cursor = conn["pageInfo"]["endCursor"]
        time.sleep(0.2) #Para evitar timeouts

# Función que pagina los comentarios de un Issue
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

# Función que pagina los comentarios de un PR
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

# Pagina los hilos de code review y comentarios de un PR
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

# Clase para la escritura incremental de los comentarios (continúa donde lo dejó y no repite comentarios aunque se hagan varias ejecuciones)
class Writer:
    def __init__(self, base: Path, dedupe: bool = True):
        self.csv_path = base.with_suffix(".csv")
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_new = not self.csv_path.exists()
        self._csv = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._w = csv.writer(self._csv)
        if self._csv_new:
            self._w.writerow([
                "repo","is_pr","issue_number","comment_type","comment_id","comment_created_at","comment_author",
                "text","comment_url","context_id","container_title","container_state","container_url","container_created_at","container_updated_at","container_labels"
            ])

        # deduplica
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
        # CSV
        self._w.writerow([
            row["repo"], row["is_pr"], row["issue_number"], row["comment_type"], row["id"], row["created_at"], row["author"],
            row["text"], row["url"], row["context_id"], row["container_title"], row["container_state"], row["container_url"], row["container_created_at"], row["container_updated_at"], ";".join(row.get("container_labels",[]))
        ])

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

# Transformación y backfill
def row_issue_body(repo_full:str, issue:Dict[str,Any]) -> Dict[str,Any]:
    return {
        "id": f"github_issuebody_{repo_full}#{issue['number']}",
        "platform":"github",
        "text": issue.get("bodyText") or "",
        "created_at": issue["createdAt"],
        "url": issue["url"],
        "repo": repo_full,
        "issue_number": issue["number"],
        "is_pr": False,
        "comment_type":"issue_body",
        "author": (issue.get("author") or {}).get("login"),
        "context_id": f"{repo_full}#issue:{issue['number']}",
        "container_title": issue.get("title"),
        "container_state": issue.get("state"),
        "container_url": issue.get("url"),
        "container_created_at": issue.get("createdAt"),
        "container_updated_at": issue.get("updatedAt"),
        "container_labels":[lb["name"] for lb in (issue.get("labels",{}) or {}).get("nodes",[])],
    }

def row_pr_body(repo_full:str, pr:Dict[str,Any]) -> Dict[str,Any]:
    return {
        "id": f"github_prbody_{repo_full}#{pr['number']}",
        "platform":"github",
        "text": pr.get("bodyText") or "",
        "created_at": pr["createdAt"],
        "url": pr["url"],
        "repo": repo_full,
        "issue_number": pr["number"],
        "is_pr": True,
        "comment_type":"pr_body",
        "author": (pr.get("author") or {}).get("login"),
        "context_id": f"{repo_full}#pr:{pr['number']}",
        "container_title": pr.get("title"),
        "container_state": pr.get("state"),
        "container_url": pr.get("url"),
        "container_created_at": pr.get("createdAt"),
        "container_updated_at": pr.get("updatedAt"),
        "container_labels":[lb["name"] for lb in (pr.get("labels",{}) or {}).get("nodes",[])],
    }

# *** Funciones para evitar desconexión por demasiadas consultas (no hacer 1 consulta por body) ***
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

# -------------------------------------------------------------------------------------------------

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

# Funciones para pedir los bodys de Issues y PRs
def fetch_issue_body(session, owner, repo, number):
    data = gql(session, ISSUE_BODY_Q, {"owner":owner,"name":repo,"number":number})
    return data["repository"]["issue"]

def fetch_pr_body(session, owner, repo, number):
    data = gql(session, PR_BODY_Q, {"owner":owner,"name":repo,"number":number})
    return data["repository"]["pullRequest"]

# Función para la extracción de comentarios desde el último extraído
def extractor_repo(session, owner, repo, writer: Writer, cutoff_ts: float, include_reviews: bool, only_initial_post: bool, max_comments_per_item: Optional[int], max_comments_per_repo: Optional[int]):
    repo_full = f"{owner}/{repo}"

    # Normaliza: 0 -> no escribir ningún comentario de ese issue/PR
    if max_comments_per_item is not None and max_comments_per_item < 0:
        max_comments_per_item = 0

    # Presupuesto de cada repo (comentarios a extraer de cada repositorio)
    repo_budget = max_comments_per_repo

    # Saber si se pueden seguir extrayendo comentarios
    def can_write_more() -> bool:
        return (repo_budget is None) or (repo_budget > 0)

    # Issues
    for it in list_issues(session, owner, repo):
        if parse_iso(it["updatedAt"]) < cutoff_ts:
            break  # ya estamos fuera de la ventana

        # Solo body: no aplica presupuesto de comentarios
        if only_initial_post:
            if parse_iso(it["createdAt"]) >= cutoff_ts:
                writer.write_unique(row_issue_body_from_node(repo_full, it))
            continue

        # Si se quieren comentarios -> Limitar por item
        if not can_write_more(): # Si se ha llegado al límite de presupuesto de comentarios
            break

        if max_comments_per_item == 0:
            continue

        count = 0
        num = it["number"]
        for issue, com in iter_issue_comments(session, owner, repo, num):
            cts = parse_iso(com["createdAt"])
            if cts >= cutoff_ts:
                if not can_write_more():
                    break
                if (max_comments_per_item is not None) and (count >= max_comments_per_item):
                    break

                row = row_issue(repo_full, num, issue, com)
                wrote = writer.write_unique(row)
                if wrote:
                    count += 1
                    if repo_budget is not None:
                        repo_budget -= 1
                if not can_write_more():
                        break

        if not can_write_more():
            break

        # No queda presupuesto y no se están descargando bodies, no seguimos con PRs
        if not only_initial_post and not can_write_more():
            return

    # PRs
    for prn in list_prs(session, owner, repo):
        if parse_iso(prn["updatedAt"]) < cutoff_ts:
            break

        # Solo body de la PR
        if only_initial_post:
            if parse_iso(prn["createdAt"]) >= cutoff_ts:
                writer.write_unique(row_pr_body_from_node(repo_full, prn))
            continue

        if not can_write_more():
            break

        # Comentarios (issue comments del PR + review comments si se pide)
        if max_comments_per_item == 0:
            continue

        count = 0
        num = prn["number"]

        # PR issue comments (pestaña Conversation)
        for pr, com in iter_pr_issue_comments(session, owner, repo, num):
            cts =parse_iso(com["createdAt"])
            if cts >= cutoff_ts:
                if not can_write_more():
                    break
                if (max_comments_per_item is not None) and (count >= max_comments_per_item):
                    break

                row = row_pr_issue(repo_full, num, pr, com)
                wrote = writer.write_unique(row)

                if wrote:
                    cont += 1
                    if repo_budget is not None:
                        repo_budget -= 1

                if not can_write_more():
                    break
        if not can_write_more():
            break

        # Review comments
        if include_reviews and (max_comments_per_item is None or count < max_comments_per_item):
            for thread_id, rc in iter_pr_review_comments(session, owner, repo, num):
                cts = parse_iso(rc["createdAt"])
                if cts >= cutoff_ts:
                    if not can_write_more():
                        break
                    if (max_comments_per_item is not None) and (count >= max_comments_per_item):
                        break

                    row = row_pr_review(repo_full, num, rc, thread_id)
                    wrote = writer.write_unique(row)
                    if wrote:
                        count += 1
                        if repo_budget is not None:
                            repo_budget -= 1
                    if not can_write_more():
                        break
            if not can_write_more():
                break

# Función para leer los repositorios desde un archivo de texto
def read_repos_file(p: Optional[str]) -> List[str]:
    if not p: return DEFAULT_REPOS
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos-file", type=str, help="Archivo con repos owner/name por línea.")
    ap.add_argument("--out-base", type=str, default="../../../data/gh_comments/train-fine_tuning/gh_comments_lastyear", help="Ruta base sin extensión.")
    ap.add_argument("--days", type=int, default=365, help="Días hacia atrás (ventana).")
    ap.add_argument("--only-initial-post", action="store_true", help="Guardar solo el body inicial de Issues/PRs (sin comentarios siguientes)")
    ap.add_argument("--max-comments-per-item", type=int, default=None, help="Máximo de comentarios por Issue/PR (0 = ninguno)")
    ap.add_argument("--include-review-comments", action="store_true", help="Incluir comentarios de code review en PRs.")
    ap.add_argument("--max-comments-per-repo", type=int, default=None, help="Límite total de comentarios a extraer por repositorio (suma de Issue + PR [+review si se incluyen].")
    args = ap.parse_args()

    repos = read_repos_file(args.repos_file)
    token = require_token()
    session = mk_session(token)
    writer = Writer(Path(args.out_base))

    cutoff_ts = (datetime.now(timezone.utc) - timedelta(days=args.days)).timestamp()

    total = 0
    for repo_full in repos:
        try:
            owner, repo = repo_full.split("/", 1)
        except ValueError:
            print(f"Repo inválido: {repo_full}"); continue
        print(f"\n==> {repo_full} (últimos {args.days} días)")
        try:
            extractor_repo(session, owner, repo, writer, cutoff_ts,
                           include_reviews=args.include_review_comments,
                           only_initial_post=args.only_initial_post,
                           max_comments_per_item=args.max_comments_per_item,
                           max_comments_per_repo=args.max_comments_per_repo)

            writer.flush()
        except Exception as e:
            print(f"  ! Error en {repo_full}: {e}")
            continue
    writer.close()
    print("\nExtracción completada.")

if __name__ == "__main__":
    main()
