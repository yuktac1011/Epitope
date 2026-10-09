"""
Laya Decision Model Service — Singleton with Thread Lock

Per SKILL.md checklist:
  - Model loaded ONCE at startup, warmed up, and guarded by a lock
  - Device auto-detected (CUDA > MPS > CPU)
  - Warmup call made after loading to compile kernels
"""

import threading
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton state — loaded once via init_laya(), reused everywhere
# ---------------------------------------------------------------------------
_agent = None
_lock = threading.Lock()
_is_ready = False


def init_laya(
    checkpoint: str = "convaiinnovations/laya",
    subfolder: Optional[str] = None,
    device: Optional[str] = None,
) -> None:
    """
    Load the Laya model once at application startup.

    Args:
        checkpoint: HuggingFace model identifier.
        subfolder: Optional subfolder for a specific checkpoint variant
                   (e.g. "multilingual" or "typed-decisions").
        device: Force a device ("cpu", "cuda", "mps"). None = auto-detect.
    """
    global _agent, _is_ready
    import laya  # import here so the module can be imported without laya installed

    logger.info("Loading Laya model from '%s' (subfolder=%s) ...", checkpoint, subfolder)

    load_kwargs = {"model_id_or_path": checkpoint}
    if subfolder:
        load_kwargs["subfolder"] = subfolder
    if device:
        load_kwargs["device"] = device

    _agent = laya.load(**load_kwargs)

    # --- Warmup: first call at a new batch shape compiles kernels ---
    logger.info("Warming up Laya with a throwaway predict call ...")
    _warmup()

    _is_ready = True
    logger.info("Laya model ready.")


def _warmup() -> None:
    """
    Make one throwaway predict call so that CUDA/MPS kernels get compiled
    before real traffic arrives.
    """
    if _agent is None:
        return

    with _lock:
        _agent.predict(
            {"text": "This is a warmup transaction for kernel compilation."},
            {
                "warmup_check": {
                    "type": "noul",
                    "instructions": "Is this text a warmup message?",
                }
            },
        )


def predict(state, questions: dict) -> dict:
    """
    Thread-safe prediction. All callers share the singleton agent
    through a lock so only one forward pass runs at a time.

    Args:
        state: str, dict, or list — the text/data to classify.
        questions: dict of question definitions (choice / score / noul).

    Returns:
        The full Laya result dict (answers, usage, etc.).

    Raises:
        RuntimeError: if the model has not been loaded yet.
    """
    if _agent is None or not _is_ready:
        raise RuntimeError(
            "Laya model is not loaded. Call init_laya() at application startup."
        )

    with _lock:
        return _agent.predict(state, questions)


def is_ready() -> bool:
    """Check whether the Laya model has been loaded and warmed up."""
    return _is_ready


def get_agent():
    """Return the raw agent for advanced usage (calibration, etc.)."""
    return _agent
