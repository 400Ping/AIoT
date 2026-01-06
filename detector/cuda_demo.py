#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""兼容舊入口，改由 helmet_cam.py 的 CUDA 模式執行。"""

import sys

from helmet_cam import main


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--mode" not in argv and "--cuda" not in argv:
        argv = ["--mode", "cuda"] + argv
    raise SystemExit(main(argv))
