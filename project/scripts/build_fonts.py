#!/usr/bin/env python3
"""
Rebuild assets/fonts/*.ttf from the @fontsource npm packages.

The TTFs are committed, so this is only needed if you want to change typeface.

    npm install @fontsource/montserrat
    python3 scripts/build_fonts.py node_modules/@fontsource
"""
import os
import sys

from fontTools.ttLib import TTFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "fonts")
JOBS = [("montserrat", "latin-700"), ("montserrat", "latin-800")]

if __name__ == "__main__":
    src_root = sys.argv[1] if len(sys.argv) > 1 else "node_modules/@fontsource"
    os.makedirs(OUT, exist_ok=True)
    for family, variant in JOBS:
        src = os.path.join(src_root, family, "files", f"{family}-{variant}-normal.woff2")
        font = TTFont(src)
        font.flavor = None            # woff2 -> plain TTF, which Pillow can read
        dst = os.path.join(OUT, f"{family}-{variant}.ttf")
        font.save(dst)
        print(f"  {dst}  {os.path.getsize(dst)} bytes")
