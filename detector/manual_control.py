#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import termios
import tty
import time

from motor_controller import MotorController


HELP_TEXT = """\
W/A/S/D 控制小車，Space 停止，Q 離開：
  w: 前進
  s: 後退
  a: 左轉
  d: 右轉
  space: 停止
  q: 離開
"""


def read_key():
    """單鍵非阻塞讀取（不需要按 Enter）。"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def main():
    mc = MotorController()
    print(HELP_TEXT)
    try:
        while True:
            key = read_key().lower()
            if key == "w":
                mc.forward()
            elif key == "s":
                mc.backward()
            elif key == "a":
                mc.left()
            elif key == "d":
                mc.right()
            elif key == " ":
                mc.stop()
            elif key == "q":
                mc.stop()
                break
            # 簡單節流避免過快重複
            time.sleep(0.05)
    except KeyboardInterrupt:
        mc.stop()
    finally:
        mc.stop()


if __name__ == "__main__":
    main()
