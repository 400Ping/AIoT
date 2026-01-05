"""GPIO compatibility layer for Raspberry Pi (RPi.GPIO / rpi-lgpio)."""

try:
    import RPi.GPIO as GPIO  # type: ignore

    GPIO_PLATFORM = "rpi"
except Exception as exc:
    raise ImportError(
        "No supported GPIO library found. Install RPi.GPIO "
        "or rpi-lgpio (recommended for Raspberry Pi 5)."
    ) from exc
