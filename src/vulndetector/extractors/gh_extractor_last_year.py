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
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Iterable, List, Optional, Tuple
from dotenv import load_dotenv, find_dotenv

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
    return s

# Función que lanza la petición contra la API de GitHub
def gql(session: requests.Session, query: str, variables: Dict[str, Any], backoff: float = 1.0) -> Dict[str, Any]:
    while True:
        r = session.post(GRAPHQL_ENDPOINT, json={"query": query, "variables": variables})
        if r.status_code == 200:
            data = r.json()
            if "errors" in data:
                msgs = " | ".join(e.get("message","") for e in data["errors"])
                if "rate limit" in msgs.lower() or "Something went wrong" in msgs:
                    time.sleep(backoff); backoff = min(backoff*1.7, 30.0); continue
                raise RuntimeError(f"GraphQL error: {msgs}")
            return data["data"]
        if r.status_code in (429, 502, 503, 504):
            time.sleep(backoff); backoff = min(backoff*1.7, 30.0); continue
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")

def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def parse_iso(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()

# Modelos de querys GraphQL
# Lista de Issues de un repositorio
ISSUES_LIST_Q = """
query Issues($owner:String!, $name:String!, $pageSize:Int!, $cursor:String) {
  repository(owner:$owner, name:$name) {
    issues(first:$pageSize, after:$cursor, orderBy:{field:UPDATED_AT, direction:DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title url state createdAt updatedAt
        labels(first:50){nodes{name}}
        comments(first:1){ totalCount } # para saber si tiene
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
        labels(first:50){nodes{name}}
        comments(first:1){ totalCount }
        reviewThreads(first:1){ totalCount }
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
    def __init__(self, base: Path):
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

    def write_row(self, row: Dict[str, Any]):
        # CSV
        self._w.writerow([
            row["repo"], row["is_pr"], row["issue_number"], row["comment_type"], row["id"], row["created_at"], row["author"],
            row["text"], row["url"], row["context_id"], row["container_title"], row["container_state"], row["container_url"], row["container_created_at"], row["container_updated_at"], ";".join(row.get("container_labels",[]))
        ])

    def flush(self):
        self._csv.flush()

    def close(self):
        self._csv.close()

# Transformación y backfill
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

# Función para la extracción de comentarios desde el último extraído
def backfill_repo(session, owner, repo, writer: Writer, cutoff_ts: float, include_reviews: bool):
    repo_full = f"{owner}/{repo}"
    cutoff_iso = iso(cutoff_ts)

    # Issues
    for it in list_issues(session, owner, repo):
        if parse_iso(it["updatedAt"]) < cutoff_ts:
            break  # ya estamos fuera de la ventana
        num = it["number"]
        for issue, com in iter_issue_comments(session, owner, repo, num):
            if parse_iso(com["createdAt"]) >= cutoff_ts:
                writer.write_row(row_issue(repo_full, num, issue, com))

    # PRs
    for prn in list_prs(session, owner, repo):
        if parse_iso(prn["updatedAt"]) < cutoff_ts:
            break
        num = prn["number"]
        for pr, com in iter_pr_issue_comments(session, owner, repo, num):
            if parse_iso(com["createdAt"]) >= cutoff_ts:
                writer.write_row(row_pr_issue(repo_full, num, pr, com))
        if include_reviews:
            for thread_id, rc in iter_pr_review_comments(session, owner, repo, num):
                if parse_iso(rc["createdAt"]) >= cutoff_ts:
                    writer.write_row(row_pr_review(repo_full, num, rc, thread_id))

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
    ap.add_argument("--include-review-comments", action="store_true", help="Incluir comentarios de code review en PRs.")
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
            backfill_repo(session, owner, repo, writer, cutoff_ts, include_reviews=args.include_review_comments)
            writer.flush()
        except Exception as e:
            print(f"  ! Error en {repo_full}: {e}")
            continue
    writer.close()
    print("\nExtracción completada.")

if __name__ == "__main__":
    main()
