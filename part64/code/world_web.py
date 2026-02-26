#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# This file is part of Fork Tales.
# Copyright (C) 2024-2025 Fork Tales Contributors
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
