from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote_plus, urlsplit, urlunsplit


def is_running_in_docker() -> bool:
    return Path("/.dockerenv").exists() or os.environ.get("RUNNING_IN_DOCKER", "").lower() in {"1", "true", "yes"}


def _build_postgres_url(user: str, password: str, host: str, port: str, db: str) -> str:
    return (
        f"postgresql://{quote_plus(user)}:{quote_plus(password)}@"
        f"{host}:{port}/{db}"
    )


def _rewrite_host_port(url: str, host: str, port: str) -> str:
    parsed = urlsplit(url)
    username = quote_plus(parsed.username or "") if parsed.username else ""
    password = quote_plus(parsed.password or "") if parsed.password else ""
    auth = ""
    if username:
        auth = username
        if parsed.password is not None:
            auth += f":{password}"
        auth += "@"

    netloc = f"{auth}{host}:{port}"
    return urlunsplit((parsed.scheme or "postgresql", netloc, parsed.path, parsed.query, parsed.fragment))


def resolve_postgres_url(*, prefer_docker: bool | None = None) -> str:
    use_docker = is_running_in_docker() if prefer_docker is None else prefer_docker

    direct = (
        os.getenv("RAG_DATABASE_URL")
        or os.getenv("DATABASE_URL")
        or ""
    ).strip()

    if direct:
        if use_docker:
            parsed = urlsplit(direct)
            if parsed.hostname in {None, "127.0.0.1", "localhost"}:
                return _rewrite_host_port(direct, "db", "5432")
        return direct

    user = os.getenv("POSTGRES_USER", "rag")
    password = os.getenv("POSTGRES_PASSWORD", "rag")
    db = os.getenv("POSTGRES_DB", "policy_rag")

    if use_docker:
        host = "db"
        port = "5432"
    else:
        host = os.getenv("POSTGRES_HOST", "127.0.0.1")
        port = os.getenv("POSTGRES_PORT", "5433")

    return _build_postgres_url(user, password, host, port, db)
