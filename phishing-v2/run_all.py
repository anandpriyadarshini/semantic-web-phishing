#!/usr/bin/env python3
"""
Top-level entry point.

Usage:
    python run_all.py

This delegates to src/pipeline/run_all.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline.run_all import main

if __name__ == "__main__":
    main()
