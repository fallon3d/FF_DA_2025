"""
Core namespace package.

Intentionally avoid importing submodules here so callers can do:
    from draft_assistant.core import sleeper, evaluation, suggestions, utils
without triggering circular imports at package import time.
"""
# No eager imports here on purpose.
__all__ = ["sleeper", "evaluation", "suggestions", "utils", "run_detection"]
