"""Shared utilities for AUTODS agents."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# LLM factory — shared by all agents
# ---------------------------------------------------------------------------

_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DEFAULT_QWEN_MODEL = "qwen-plus"
_DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def build_chat_llm(
    model: Optional[str] = None,
    temperature: float = 0.0,
):
    """Return the best available ChatOpenAI-compatible LLM client.

    Priority:
      1. Qwen (DashScope) — if DASHSCOPE_API_KEY is set
      2. OpenAI            — if OPENAI_API_KEY is set
      3. None              — rule-based fallback

    The caller can pass a specific ``model`` name; when omitted the function
    uses QWEN_MODEL / OPENAI_MODEL env vars, falling back to sensible defaults.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        return None

    # ── 1. Qwen (DashScope) ──────────────────────────────────────────────────
    dashscope_key = os.getenv("DASHSCOPE_API_KEY", "")
    if dashscope_key and "placeholder" not in dashscope_key.lower():
        # If the caller passed an OpenAI-specific model name (gpt-*, o1-*, o3-*),
        # replace it with the Qwen model — those names don't exist on DashScope.
        _is_openai_name = model and (
            model.startswith("gpt-") or model.startswith("o1") or model.startswith("o3")
        )
        qwen_model = (
            os.getenv("QWEN_MODEL", _DEFAULT_QWEN_MODEL)
            if (model is None or _is_openai_name)
            else model
        )
        try:
            llm = ChatOpenAI(
                model=qwen_model,
                temperature=temperature,
                api_key=dashscope_key,
                base_url=_QWEN_BASE_URL,
            )
            print(f"[LLM] Using Qwen ({qwen_model})")
            return llm
        except Exception as exc:
            print(f"[LLM] Qwen init failed: {exc}")

    # ── 2. OpenAI ────────────────────────────────────────────────────────────
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key and "placeholder" not in openai_key.lower():
        openai_model = model or os.getenv("OPENAI_MODEL", _DEFAULT_OPENAI_MODEL)
        base_url = os.getenv("OPENAI_BASE_URL") or None
        try:
            llm = ChatOpenAI(
                model=openai_model,
                temperature=temperature,
                api_key=openai_key,
                base_url=base_url,
            )
            print(f"[LLM] Using OpenAI ({openai_model})")
            return llm
        except Exception as exc:
            print(f"[LLM] OpenAI init failed: {exc}")

    return None


def load_project_env(anchor_file: str | Path) -> None:
    """Load the .env file located in the same directory as anchor_file (or
    any parent directory up to the filesystem root).  Falls back silently
    when python-dotenv is not installed or no .env file is found."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    search = Path(anchor_file).resolve().parent
    for directory in [search, *search.parents]:
        candidate = directory / ".env"
        if candidate.exists():
            load_dotenv(candidate, override=False)
            return
    # No .env found — still call load_dotenv so any OS-level env vars are used
    load_dotenv(override=False)


def reexec_with_project_venv(script_file: str | Path) -> None:
    """Re-execute the current script inside the project's virtual environment.

    This is a CLI helper used when a script is run directly (``__name__ ==
    '__main__'``).  It is a no-op when the script is imported as a module or
    when the project does not have a venv directory alongside it.
    """
    script_path = Path(script_file).resolve()
    for venv_name in ("venv", ".venv", "env"):
        venv_python = script_path.parent / venv_name / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
            os.execv(str(venv_python), [str(venv_python)] + sys.argv)
            return  # unreachable if execv succeeds


def json_default(obj):
    """JSON serialiser fallback that handles numpy scalar types.

    Pass this as the ``default`` argument to ``json.dump`` / ``json.dumps``
    so that numpy integers, floats, and booleans produced by pandas/sklearn
    are serialised without raising ``TypeError``.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return str(obj)
