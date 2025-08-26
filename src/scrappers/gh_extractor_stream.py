#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
No existe “stream” en tiempo real en GitHub como en Reddit, así que implemento un polling inteligente:
consulta cada X segundos los issues/PRs más recientemente actualizados y recoge comentarios nuevos desde
el último checkpoint por repo (tiempo e ID). Si se corta el proceso, al reiniciar hace catch-up y continúa.

“Streaming” por polling de comentarios nuevos de GitHub con checkpoint por repo.
- Al iniciar: catch-up desde el último checkpoint (o tiempo inicial).
- Luego: bucle que consulta issues/PRs actualizados recientemente y extrae
  comentarios con createdAt > checkpoint.
- Guarda CSV/JSONL y checkpoints {repo}.json (last_ts, last_id).

Uso:
  export GITHUB_TOKEN=ghp_xxx
  python gh_stream_with_checkpoint.py --repos-file repos.txt --out-base ../../../data/gh_comments_stream --interval 30
"""

import os, sys, json, time, signal, csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import requests

GRAPHQL_ENDPOINT = "https://api.github.com/graphql"
PAGE_SIZE = 100

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

def require_token() -> str:
    tok = os.getenv("GITHUB_TOKEN")
    if not tok:
        print("ERROR: define GITHUB_TOKEN.")
        sys.exit(1)
    return tok

def mk_session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": "application/json",
        "User-Agent": "gh-comments-stream/1.0"
    })
    return s

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

# ── Queries ──────────────────────────────────────────────────────────────────
ISSUES_LIST_Q = """
query Issues($owner:String!, $name:String!, $pageSize:Int!, $cursor:String) {
  repository(owner:$owner, name:$name) {
    issues(first:$pageSize, after:$cursor, orderBy:{field:UPDATED_AT, direction:DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes { number title url state createdAt updatedAt labels(first:50){nodes{name}} comments(first:1){ totalCount } }
    }
  }
}
"""

PRS_LIST_Q = """
query PRs($owner:String!, $name:String!, $pageSize:Int!, $cursor:String) {
  repository(owner:$owner, name:$name) {
    pullRequests(first:$pageSize, after:$cursor, orderBy:{field:UPDATED_AT, direction:DESC}, states:[OPEN, MERGED, CLOSED]) {
      pageInfo { hasNextPage endCursor }
      nodes { number title url state createdAt updatedAt mergedAt labels(first:50){nodes{name}}
              comments(first:1){ totalCount } reviewThreads(first:1){ totalCount } }
    }
  }
}
"""

ISSUE_COMMENTS_Q = """
query IssueComments($owner:String!, $name:String!, $number:Int!, $pageSize:Int!, $cursor:String) {
  repository(owner:$owner, name:$name) {
    issue(number:$number) {
      title url state createdAt updatedAt labels(first:50){nodes{name}}
      comments(first:$pageSize, after:$cursor) {
        pageInfo { hasNextPage endCursor }
        nodes { id url bodyText createdAt updatedAt isMinimized author{login} }
      }
    }
  }
}
"""

PR_COMMENTS_Q = """
query PRComments($owner:String!, $name:String!, $number:Int!, $pageSize:Int!, $cursor:String) {
  repository(owner:$owner, name:$name) {
    pullRequest(number:$number) {
      title url state createdAt updatedAt mergedAt labels(first:50){nodes{name}}
      comments(first:$pageSize, after:$cursor) {
        pageInfo { hasNextPage endCursor }
        nodes { id url bodyText createdAt updatedAt isMinimized author{login} }
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
          comments(first:100) { nodes { id url bodyText createdAt updatedAt author{login} } }
        }
      }
    }
  }
}
"""

# ── Iteradores ────────────────────────────────────────────────────────────────
def list_issues(session, owner, repo):
    cursor = None
    while True:
        data = gql(session, ISSUES_LIST_Q, {"owner":owner,"name":repo,"pageSize":PAGE_SIZE,"cursor":cursor})
        conn = data["repository"]["issues"]
        for n in conn["nodes"]:
            yield n
        if not conn["pageInfo"]["hasNextPage"]: break
        cursor = conn["pageInfo"]["endCursor"]

def list_prs(session, owner, repo):
    cursor = None
    while True:
        data = gql(session, PRS_LIST_Q, {"owner":owner,"name":repo,"pageSize":PAGE_SIZE,"cursor":cursor})
        conn = data["repository"]["pullRequests"]
        for n in conn["nodes"]:
            yield n
        if not conn["pageInfo"]["hasNextPage"]: break
        cursor = conn["pageInfo"]["endCursor"]

def iter_issue_comments(session, owner, repo, number):
    cursor = None
    while True:
        data = gql(session, ISSUE_COMMENTS_Q, {"owner":owner,"name":repo,"number":number,"pageSize":PAGE_SIZE,"cursor":cursor})
        issue = data["repository"]["issue"]
        if not issue: break
        for n in issue["comments"]["nodes"]:
            yield issue, n
        if not issue["comments"]["pageInfo"]["hasNextPage"]: break
        cursor = issue["comments"]["pageInfo"]["endCursor"]

def iter_pr_issue_comments(session, owner, repo, number):
    cursor = None
    while True:
        data = gql(session, PR_COMMENTS_Q, {"owner":owner,"name":repo,"number":number,"pageSize":PAGE_SIZE,"cursor":cursor})
        pr = data["repository"]["pullRequest"]
        if not pr: break
        for n in pr["comments"]["nodes"]:
            yield pr, n
        if not pr["comments"]["pageInfo"]["hasNextPage"]: break
        cursor = pr["comments"]["pageInfo"]["endCursor"]

def iter_pr_review_comments(session, owner, repo, number):
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

# ── Escritura y checkpoint ───────────────────────────────────────────────────
class Writer:
    def __init__(self, base: Path):
        self.csv_path = base.with_suffix(".csv"); self.jsonl_path = base.with_suffix(".jsonl")
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.csv_path.exists()
        self._csv = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._w = csv.writer(self._csv)
        if new:
            self._w.writerow([
                "repo","is_pr","issue_number","comment_type","comment_id","comment_created_at","comment_author",
                "text","comment_url","context_id","container_title","container_state","container_url","container_created_at","container_updated_at","container_labels"
            ])
        self._jsonl = open(self.jsonl_path, "a", encoding="utf-8")

    def write_row(self, row: Dict[str, Any]):
        self._w.writerow([
            row["repo"], row["is_pr"], row["issue_number"], row["comment_type"], row["id"], row["created_at"], row["author"],
            row["text"], row["url"], row["context_id"], row["container_title"], row["container_state"], row["container_url"], row["container_created_at"], row["container_updated_at"], ";".join(row.get("container_labels",[]))
        ])
        self._jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")

    def flush(self):
        self._csv.flush(); self._jsonl.flush()

    def close(self):
        self._csv.close(); self._jsonl.close()

def ckpt_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True); return path

def ckpt_path(base_dir: Path, repo_full: str) -> Path:
    safe = repo_full.replace("/", "__")
    return base_dir / f"{safe}.json"

def load_ckpt(base_dir: Path, repo_full: str) -> Tuple[float, Optional[str]]:
    p = ckpt_path(base_dir, repo_full)
    if not p.exists(): return (0.0, None)
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return (float(d.get("last_comment_created_utc", 0.0)), d.get("last_comment_id"))
    except Exception:
        return (0.0, None)

def save_ckpt(base_dir: Path, repo_full: str, ts: float, cid: str):
    p = ckpt_path(base_dir, repo_full)
    data = {
        "repo": repo_full,
        "last_comment_created_utc": float(ts),
        "last_comment_created_iso": iso(ts),
        "last_comment_id": cid,
        "updated_at": iso(time.time()),
    }
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

# ── Transformación ───────────────────────────────────────────────────────────
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

# ── Escaneo de “novedades” ───────────────────────────────────────────────────
def scan_repo_once(session, repo_full: str, writer: Writer, last_ts: float, include_reviews: bool, lookback_minutes: int = 10) -> float:
    """
    Escanea issues/PRs cuyo updatedAt sea reciente (usamos orden DESC y cortamos
    cuando updatedAt < last_ts - margen). Extrae comentarios con createdAt > last_ts.
    Devuelve el nuevo last_ts si encontró más reciente, si no devuelve el mismo.
    """
    owner, repo = repo_full.split("/", 1)
    repo_latest_ts = last_ts
    cutoff = last_ts - (lookback_minutes * 60)

    # Issues
    for it in list_issues(session, owner, repo):
        if parse_iso(it["updatedAt"]) < cutoff:
            break
        num = it["number"]
        for issue, com in iter_issue_comments(session, owner, repo, num):
            cts = parse_iso(com["createdAt"])
            if cts > last_ts:
                writer.write_row(row_issue(repo_full, num, issue, com))
                if cts > repo_latest_ts: repo_latest_ts = cts

    # PRs
    for prn in list_prs(session, owner, repo):
        if parse_iso(prn["updatedAt"]) < cutoff:
            break
        num = prn["number"]
        for pr, com in iter_pr_issue_comments(session, owner, repo, num):
            cts = parse_iso(com["createdAt"])
            if cts > last_ts:
                writer.write_row(row_pr_issue(repo_full, num, pr, com))
                if cts > repo_latest_ts: repo_latest_ts = cts
        if include_reviews:
            for thread_id, rc in iter_pr_review_comments(session, owner, repo, num):
                cts = parse_iso(rc["createdAt"])
                if cts > last_ts:
                    writer.write_row(row_pr_review(repo_full, num, rc, thread_id))
                    if cts > repo_latest_ts: repo_latest_ts = cts

    return repo_latest_ts

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
    import argparse
    ap = argparse.ArgumentParser(description="Polling continuo de comentarios nuevos (con checkpoint por repo).")
    ap.add_argument("--repos-file", type=str, help="Archivo con repos owner/name por línea.")
    ap.add_argument("--out-base", type=str, default="../../../data/gh_comments_stream", help="Ruta base de salida SIN extensión.")
    ap.add_argument("--checkpoint-dir", type=str, default="../../../data/checkpoints/github_stream", help="Directorio de checkpoints.")
    ap.add_argument("--interval", type=int, default=30, help="Segundos entre rondas de polling.")
    ap.add_argument("--lookback-minutes", type=int, default=10, help="Margen al pasado para no perder eventos entre polls.")
    ap.add_argument("--include-review-comments", action="store_true", help="Incluir comentarios de code review en PRs.")
    args = ap.parse_args()

    token = require_token()
    session = mk_session(token)
    repos = read_repos_file(args.repos_file)
    out_base = Path(args.out_base)
    ckpt_base = Path(args.checkpoint_dir); ckpt_dir(ckpt_base)
    writer = Writer(out_base)

    # Cargar checkpoints
    last_map: Dict[str, float] = {}
    for r in repos:
        ts, _ = load_ckpt(ckpt_base, r)
        last_map[r] = ts or 0.0

    stop = False
    def handle(sig, frame):
        nonlocal stop
        stop = True
        print("\nSeñal capturada; guardando checkpoints…")
        for r in repos:
            ts = last_map.get(r, 0.0)
            if ts > 0:
                save_ckpt(ckpt_base, r, ts, f"ts_{int(ts)}")
        writer.flush(); writer.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)

    print("Iniciando polling continuo… (Ctrl+C para salir)")
    while not stop:
        round_start = time.time()
        try:
            for r in repos:
                try:
                    new_ts = scan_repo_once(session, r, writer, last_map.get(r, 0.0), include_reviews=args.include_review_comments, lookback_minutes=args.lookback_minutes)
                    if new_ts > last_map.get(r, 0.0):
                        last_map[r] = new_ts
                        save_ckpt(ckpt_base, r, new_ts, f"ts_{int(new_ts)}")
                except Exception as e:
                    print(f"[{r}] error: {e}")
            writer.flush()
        except Exception as e:
            print(f"[loop] error: {e}")
        # dormir hasta siguiente ronda
        elapsed = time.time() - round_start
        sleep_for = max(1, args.interval - int(elapsed))
        time.sleep(sleep_for)

if __name__ == "__main__":
    main()
