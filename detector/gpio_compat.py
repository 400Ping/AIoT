"""GPIO compatibility layer for Jetson.GPIO / RPi.GPIO."""

try:
    import Jetson.GPIO as GPIO  # type: ignore

    GPIO_PLATFORM = "jetson"
except Exception:
    try:
        import RPi.GPIO as GPIO  # type: ignore

        GPIO_PLATFORM = "rpi"
    except Exception as exc:
        raise ImportError(
            "No supported GPIO library found. Install Jetson.GPIO on Jetson "
            "or RPi.GPIO on Raspberry Pi."
        ) from exc

