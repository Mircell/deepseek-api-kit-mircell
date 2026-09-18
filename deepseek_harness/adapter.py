"""Backward-compatibility shim.

The DeepSeek adapter logic now lives in :mod:`deepseek_harness.provider`,
built on top of ``fastapi-openai-compat``. This module is kept only so that
existing imports (e.g. ``from deepseek_harness.adapter import DeepSeekAdapter``)
continue to resolve.
"""

from __future__ import annotations

from common.api import DeepSeekAPI as DeepSeekAdapter  # noqa: F401
from .provider import api, list_models, run_completion  # noqa: F401

__all__ = ["DeepSeekAdapter", "api", "list_models", "run_completion"]