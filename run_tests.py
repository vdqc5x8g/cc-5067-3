#!/usr/bin/env python3
"""Dependency-free test runner for the mbdyn suite (pytest not required)."""

import glob
import importlib
import os
import sys
import time
import traceback

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py")))
    total = 0
    failed = 0
    t_all = time.time()
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        mod = importlib.import_module(name)
        for attr in sorted(dir(mod)):
            if not attr.startswith("test_"):
                continue
            total += 1
            t0 = time.time()
            try:
                getattr(mod, attr)()
                print(f"PASS {name}.{attr} ({time.time() - t0:.2f}s)")
            except Exception:
                failed += 1
                print(f"FAIL {name}.{attr}")
                traceback.print_exc()
    print(f"\n{total - failed}/{total} passed in {time.time() - t_all:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
