import os
import sys
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chirrp.backend.dbo.db_config import resolve_postgres_url


load_dotenv()


def resolve_dsn() -> str:
    return resolve_postgres_url()


def wait_for_db(dsn: str, retries: int = 40, sleep_sec: int = 2) -> None:
    last_err = None
    for _ in range(retries):
        try:
            with psycopg.connect(dsn) as conn, conn.cursor() as cur:
                cur.execute("SELECT 1")
            return
        except Exception as e:
            last_err = e
            time.sleep(sleep_sec)
    raise RuntimeError(f"Database did not become ready: {last_err}")


def run_sql_files(dsn: str, files: list[Path]) -> None:
    with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
        for file in files:
            sql = file.read_text(encoding="utf-8")
            cur.execute(sql)
            print(f"Applied SQL: {file}")


def ensure_rag_schema_compat(dsn: str) -> None:
        migration_sql = """
        ALTER TABLE IF EXISTS doc_chunks
            ADD COLUMN IF NOT EXISTS heading TEXT,
            ADD COLUMN IF NOT EXISTS section_path TEXT;

        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = 'doc_chunks'
                    AND column_name = 'embedding'
            ) THEN
                BEGIN
                    ALTER TABLE doc_chunks ALTER COLUMN embedding TYPE vector(1536);
                EXCEPTION WHEN others THEN
                    TRUNCATE TABLE doc_chunks;
                    ALTER TABLE doc_chunks ALTER COLUMN embedding TYPE vector(1536);
                END;
            END IF;
        END $$;
        """

        with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
                cur.execute(migration_sql)
        print("Applied RAG schema compatibility migration.")


def main() -> None:
    curr_dir = Path(__file__).parent
    rag_dir = Path(__file__).parents[2] / "rag_pipeline"
    dsn = resolve_dsn()

    sql_files = [
        rag_dir / "sql" / "001_init.sql",
        rag_dir / "sql" / "002_tables.sql",
        curr_dir / "sql_tables" / "create_pinn_runs_table.sql",
        curr_dir / "sql_tables" / "create_sessions_tables.sql",
        curr_dir / "sql_tables" / "create_participants_tables.sql",
    ]

    missing = [str(p) for p in sql_files if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing SQL files: {missing}")

    wait_for_db(dsn)
    run_sql_files(dsn, sql_files)
    ensure_rag_schema_compat(dsn)
    print("Database initialization complete.")


if __name__ == "__main__":
    main()
