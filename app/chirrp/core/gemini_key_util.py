from __future__ import annotations

import os
from pathlib import Path


def resolve_gemini_api_key() -> str:
    chirrp_dir = Path(__file__).resolve().parents[1]
    root_dir = Path(__file__).resolve().parents[2]
    candidate_files = [
        chirrp_dir / "gemini_api.txt",
        root_dir / "gemini_api.txt",
    ]

    for api_file in candidate_files:
        if api_file.is_file():
            key = api_file.read_text(encoding="utf-8").strip()
            if key:
                return key

    return os.environ.get("GEMINI_API_KEY", "").strip()