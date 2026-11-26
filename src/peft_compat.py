"""Compatibility guard for environments (like Colab) where peft and torchao versions mismatch."""

from __future__ import annotations

import sys


def patch_peft_torchao() -> None:
    """Prevent peft's torchao dispatcher from raising ImportError on incompatible torchao."""
    try:
        import peft.import_utils as import_utils

        orig = import_utils.is_torchao_available

        def _safe_is_torchao_available() -> bool:
            try:
                return orig()
            except ImportError:
                return False

        import_utils.is_torchao_available = _safe_is_torchao_available
        try:
            import peft.tuners.lora.torchao as lora_torchao

            lora_torchao.is_torchao_available = _safe_is_torchao_available
        except Exception:
            pass
        for mod in list(sys.modules.values()):
            if mod is not None and getattr(mod, "__name__", "").startswith("peft."):
                if hasattr(mod, "is_torchao_available"):
                    setattr(mod, "is_torchao_available", _safe_is_torchao_available)
    except Exception:
        pass
