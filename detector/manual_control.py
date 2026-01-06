#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""兼容舊入口，改由 car_main.py 負責 WASD 控制。"""

from car_main import main


if __name__ == "__main__":
    raise SystemExit(main())
