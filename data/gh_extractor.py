#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scrape de comentarios PUROS (issue comments) de múltiples URLs de GitHub (Issue/PR) vía GraphQL.

- Lee las URLs desde un archivo .txt (una URL por línea). Ejemplos:
    https://github.com/owner/repo/issues/123
    https://github.com/owner/repo/pull/456

- Extrae ÚNICAMENTE "issue comments" (los de la pestaña de conversación). NO extrae
  comentarios de code review (threads en el diff).

- Token:
    * Por defecto lee ../token.txt (como tu flujo original)
    * Alternativa: variable de entorno GITHUB_TOKEN

- Salida:
    * CSV y JSONL (orient=records, lines=True) consolidados en una sola ruta base (--out-base)
    * Texto SIN normalizar (tal cual devuelve GitHub, campo bodyText)

Uso:
    python gh_scrape_graphql_urls_pure.py --urls-file urls.txt --out-base ../data/github_comments
    python gh_scrape_graphql_urls_pure.py --urls-file urls.txt --out-base ../data/github_comments --since 2024-01-01T00:00:00Z

Requisitos:
    pip install requests pandas urllib3
"""

import os
import sys
import json
import time
import argparse
from typing import Dict, Any, Tuple, Optional, List
from urllib.parse import urlparse
from datetime import datetime, timezone
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

GRAPHQL_ENDPOINT = "https://api.github.com/graphql"
ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"


# -------------------------
# Utilidades
# -------------------------

def load_token(filepath: str = "../token.txt") -> Optional[str]:
    """Intenta leer el token de ../token.txt; si no existe, usa env GITHUB_TOKEN."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                t = f.read().strip()
                if t:
                    return t
        except Exception:
            pass
    return os.getenv("GITHUB_TOKEN")


def parse_github_url(url: str) -> Tuple[str, str, str, int]:
    """
    Devuelve (owner, repo, kind, number) donde kind ∈ {"issues", "pull"}.
    Lanza ValueError si la URL no es válida.
    """
    p = urlparse(url)
    if p.netloc.lower() != "github.com":
        raise ValueError(f"Host no soportado: {p.netloc}")
    parts = [x for x in p.path.strip("/").split("/") if x]
    if len(parts) < 4:
        raise ValueError(f"Formato de URL no reconocido: {url}")
    owner, repo, kind, number_s = parts[0], parts[1], parts[2], parts[3]
    if kind not in ("issues", "pull"):
        raise ValueError(f"La URL debe apuntar a /issues/<n> o /pull/<n>; recibido /{kind}/")
    try:
        number = int(number_s)
    except ValueError:
        raise ValueError(f"Número de issue/PR inválido en la URL: {number_s}")
    return owner, repo, kind, number


def mk_session(token: str) -> requests.Session:
    """Crea sesión con retries/backoff y cabeceras para GraphQL."""
    session = requests.Session()
    retry = Retry(
        total=8,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
        "User-Agent": "vuln-detector-graphql/1.0"
    })
    return session


def gql(session: requests.Session, query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    """Llama a GraphQL con reintentos/backoff ante errores transitorios/rate limit."""
    backoff = 1.0
    while True:
        r = session.post(GRAPHQL_ENDPOINT, json={"query": query, "variables": variables})
        if r.status_code == 200:
            data = r.json()
            if "errors" in data:
                msgs = " | ".join(e.get("message", "") for e in data["errors"])
                if "Something went wrong while executing your query" in msgs or "rate limit" in msgs.lower():
                    time.sleep(backoff)
                    backoff = min(backoff * 1.7, 30.0)
                    continue
                raise RuntimeError(f"GraphQL error: {msgs}")
            return data["data"]
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(backoff)
            backoff = min(backoff * 1.7, 30.0)
            continue
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")


def parse_iso(s: str) -> datetime:
    return datetime.strptime(s, ISO_FMT).replace(tzinfo=timezone.utc)


# -------------------------
# Queries GraphQL
# -------------------------

ISSUE_QUERY = """
query IssueComments($owner: String!, $name: String!, $number: Int!, $pageSize: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    issue(number: $number) {
      id
      url
      number
      title
      state
      createdAt
      closedAt
      labels(first: 50) { nodes { name } }
      comments(first: $pageSize, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id
          url
          bodyText
          createdAt
          updatedAt
          isMinimized
          author { login __typename }
        }
      }
    }
  }
}
"""

PR_COMMENTS_QUERY = """
query PRComments($owner: String!, $name: String!, $number: Int!, $pageSize: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      id
      url
      number
      title
      state
      createdAt
      mergedAt
      labels(first: 50) { nodes { name } }
      comments(first: $pageSize, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id
          url
          bodyText
          createdAt
          updatedAt
          isMinimized
          author { login __typename }
        }
      }
    }
  }
}
"""


# -------------------------
# Iteradores de paginación
# -------------------------

def iter_issue_comments(session: requests.Session, owner: str, repo: str, number: int, page_size: int = 100):
    cursor = None
    while True:
        data = gql(session, ISSUE_QUERY, {
            "owner": owner, "name": repo, "number": number,
            "pageSize": page_size, "cursor": cursor
        })
        issue = data["repository"]["issue"]
        if not issue:
            break
        conn = issue["comments"]
        for node in conn["nodes"]:
            yield {"container": issue, "node": node}
        if not conn["pageInfo"]["hasNextPage"]:
            break
        cursor = conn["pageInfo"]["endCursor"]


def iter_pr_issue_comments(session: requests.Session, owner: str, repo: str, number: int, page_size: int = 100):
    cursor = None
    while True:
        data = gql(session, PR_COMMENTS_QUERY, {
            "owner": owner, "name": repo, "number": number,
            "pageSize": page_size, "cursor": cursor
        })
        pr = data["repository"]["pullRequest"]
        if not pr:
            break
        conn = pr["comments"]
        for node in conn["nodes"]:
            yield {"container": pr, "node": node}
        if not conn["pageInfo"]["hasNextPage"]:
            break
        cursor = conn["pageInfo"]["endCursor"]


# -------------------------
# Transformación a filas normalizadas (sin tocar el texto)
# -------------------------

def to_row_issue(node: Dict[str, Any], issue: Dict[str, Any], repo_full: str, number: int, source_url: str) -> Dict[str, Any]:
    return {
        "id": f"github_issuecomment_{node['id']}",
        "platform": "github",
        "text": node.get("bodyText") or "",
        "created_at": node["createdAt"],
        "url": node["url"],
        "repo": repo_full,
        "issue_number": number,
        "is_pr": False,
        "comment_type": "issue_comment",
        "author": (node.get("author") or {}).get("login"),
        "context_id": f"{repo_full}#issue:{number}",
        "container_title": issue.get("title"),
        "container_state": issue.get("state"),
        "container_labels": [lb["name"] for lb in (issue.get("labels", {}) or {}).get("nodes", [])],
        "container_url": issue.get("url"),
        "container_created_at": issue.get("createdAt"),
        "container_closed_at": issue.get("closedAt"),
        "source_url": source_url,
    }


def to_row_pr_issue(node: Dict[str, Any], pr: Dict[str, Any], repo_full: str, number: int, source_url: str) -> Dict[str, Any]:
    return {
        "id": f"github_pr_issuecomment_{node['id']}",
        "platform": "github",
        "text": node.get("bodyText") or "",
        "created_at": node["createdAt"],
        "url": node["url"],
        "repo": repo_full,
        "issue_number": number,
        "is_pr": True,
        "comment_type": "issue_comment",
        "author": (node.get("author") or {}).get("login"),
        "context_id": f"{repo_full}#pr:{number}",
        "container_title": pr.get("title"),
        "container_state": pr.get("state"),
        "container_labels": [lb["name"] for lb in (pr.get("labels", {}) or {}).get("nodes", [])],
        "container_url": pr.get("url"),
        "container_created_at": pr.get("createdAt"),
        "container_closed_at": pr.get("mergedAt") or None,
        "source_url": source_url,
    }


# -------------------------
# Main
# -------------------------

def read_urls(file_path: str, max_urls: Optional[int] = None) -> List[str]:
    """Lee URLs de un .txt; ignora líneas vacías y comentarios (#)."""
    urls = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
            if max_urls and len(urls) >= max_urls:
                break
    return urls


def main():
    ap = argparse.ArgumentParser(description="Scraper de comentarios PUROS (issue comments) con GraphQL para múltiples URLs (issue/PR).")
    ap.add_argument("--urls-file", required=True, help="Ruta a .txt con una URL por línea (issues o PRs).")
    ap.add_argument("--out-base", default="../data/github_comments", help="Ruta base de salida SIN extensión (crea .csv y .jsonl).")
    ap.add_argument("--since", default=None, help="ISO UTC (YYYY-MM-DDTHH:MM:SSZ) para filtrar por fecha mínima de creación (post-filtrado).")
    ap.add_argument("--page-size", type=int, default=100, help="Tamaño de página para GraphQL (por defecto 100).")
    ap.add_argument("--max-urls", type=int, default=None, help="Máximo de URLs a procesar (debug).")
    args = ap.parse_args()

    token = load_token()
    if not token:
        print("No se encontró token. Añádelo a ../token.txt o define GITHUB_TOKEN.")
        sys.exit(1)

    urls = read_urls(args.urls_file, args.max_urls)
    if not urls:
        print("El archivo de URLs está vacío o no válido.")
        sys.exit(1)

    session = mk_session(token)
    rows = []
    seen_ids = set()
    total_urls = len(urls)
    print(f"Procesando {total_urls} URLs...")

    for idx, url in enumerate(urls, 1):
        try:
            owner, repo, kind, number = parse_github_url(url)
        except Exception as e:
            print(f"[{idx}/{total_urls}] URL inválida: {url} ({e})")
            continue

        repo_full = f"{owner}/{repo}"
        count_before = len(rows)

        try:
            if kind == "issues":
                for item in iter_issue_comments(session, owner, repo, number, page_size=args.page_size):
                    row = to_row_issue(item["node"], item["container"], repo_full, number, url)
                    if row["id"] not in seen_ids:
                        rows.append(row); seen_ids.add(row["id"])
            else:  # PR
                for item in iter_pr_issue_comments(session, owner, repo, number, page_size=args.page_size):
                    row = to_row_pr_issue(item["node"], item["container"], repo_full, number, url)
                    if row["id"] not in seen_ids:
                        rows.append(row); seen_ids.add(row["id"])
        except Exception as e:
            print(f"[{idx}/{total_urls}] Error al procesar {url}: {e}")
            continue

        added = len(rows) - count_before
        print(f"[{idx}/{total_urls}] {url} → +{added} comentarios")

    if not rows:
        print("No se encontraron comentarios en las URLs dadas.")
        sys.exit(0)

    # Post-filtrado por fecha si --since (no altera el texto)
    if args.since:
        try:
            since_dt = parse_iso(args.since)
            rows = [r for r in rows if parse_iso(r["created_at"]) >= since_dt]
        except Exception as e:
            print(f"WARNING: --since inválido ({e}). Se ignora el filtro.")

    # Orden por fecha
    rows.sort(key=lambda r: r["created_at"])

    # DataFrame y guardado
    df = pd.DataFrame(rows)
    base = args.out_base
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    csv_path = f"{base}.csv"
    jsonl_path = f"{base}.jsonl"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)

    print(f"\nOK: {len(df)} comentarios únicos guardados en:")
    print(f" - {csv_path}")
    print(f" - {jsonl_path}")

    # Muestra de ejemplo
    try:
        pd.set_option('display.max_colwidth', 120)
        print("\nMuestra aleatoria:")
        print(df[['id', 'created_at', 'repo', 'issue_number', 'text']].sample(min(5, len(df)), random_state=42))
    except Exception:
        pass


if __name__ == "__main__":
    main()
