#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _load_world_web_package() -> Any:
    part_root = Path(__file__).resolve().parent.parent
    code_dir = Path(__file__).resolve().parent

    for candidate in (part_root, code_dir):
        candidate_text = str(candidate)
        if candidate_text not in sys.path:
            sys.path.insert(0, candidate_text)
        cached = sys.modules.get("code")
        if cached is not None and not hasattr(cached, "__path__"):
            sys.modules.pop("code", None)
        try:
            from code import world_web as package  # type: ignore

            return package
        except Exception:
            pass

    import world_web as package  # type: ignore

    return package


_PACKAGE = _load_world_web_package()
for _name in dir(_PACKAGE):
    if _name.startswith("__"):
        continue
    globals()[_name] = getattr(_PACKAGE, _name)

main = getattr(_PACKAGE, "main")

if __name__ == "__main__":
    sys.exit(main())
