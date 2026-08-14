"""Build Part B tables, comparison figures, and fact sheets from saved CSVs only."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.presentation import build_presentation_outputs

ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> None:
    paths = build_presentation_outputs(ROOT)
    for name, path in sorted(paths.items()):
        print(f"{name}: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
