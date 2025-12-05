#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import readchar  # pip install readchar
from motor_controller import MotorController


def main():
    # 只負責馬達控制，不做偵測、不開鏡頭
    motors = MotorController(default_move_time=0.3)

    print("WASD 控車，空白鍵停車，q 離開。")
    print("請把終端機視窗框選在前景，直接按鍵即可。")

    try:
        while True:
            ch = readchar.readkey()

            if ch.lower() == "w":
                print("[KEY] w → forward")
                motors.forward_continuous() if hasattr(motors, "forward_continuous") else motors.forward()

            elif ch.lower() == "s":
                print("[KEY] s → backward")
                motors.backward_continuous() if hasattr(motors, "backward_continuous") else motors.backward()

            elif ch.lower() == "a":
                print("[KEY] a → turn left")
                motors.turn_left_continuous() if hasattr(motors, "turn_left_continuous") else motors.left()

            elif ch.lower() == "d":
                print("[KEY] d → turn right")
                motors.turn_right_continuous() if hasattr(motors, "turn_right_continuous") else motors.right()

            elif ch == " ":
                print("[KEY] space → stop")
                motors.stop()

            elif ch.lower() == "q":
                print("[KEY] q → quit")
                motors.stop()
                break

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt, exiting...")

    finally:
        print("Shutting down motor controller...")
        try:
            motors.stop()
        except Exception:
            pass

        # 如果你的 MotorController 有 cleanup() 的話就呼叫，沒有也不會壞
        try:
            motors.cleanup()
        except AttributeError:
            pass


if __name__ == "__main__":
    main()
