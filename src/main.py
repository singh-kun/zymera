"""Compatibility entry point: 'python src/main.py --phase ...' still works.

Prefer the installed CLI: pip install -e .  ->  zymera generate ...
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from zymera.cli import main

if __name__ == "__main__":
    sys.exit(main())
