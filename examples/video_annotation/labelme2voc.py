#!/usr/bin/env python

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    target = (
        Path(__file__).resolve().parent.parent
        / "semantic_segmentation"
        / "labelme2voc.py"
    )
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
