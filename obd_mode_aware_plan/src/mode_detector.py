import pandas as pd

from .config import (
    STOPPED_SPEED_KMH,
    ACCEL_THRESHOLD_MPS2,
    DECEL_THRESHOLD_MPS2,
    STRONG_DECEL_THRESHOLD_MPS2,
    STARTUP_GUARD_SECONDS,
)


def classify_row(row):
    """Return a simple operating mode for one preprocessed OBD sample."""

    if not bool(row.get("ENGINE_ON", False)):
        return "ENGINE_OFF"

    speed = row.get("SPEED_SMOOTH")
    accel = row.get("ACCELERATION_MPS2")
    seconds_since_start = row.get("SECONDS_SINCE_START")

    if pd.isna(speed):
        return "UNKNOWN_TRANSITION"

    if (
        speed <= STOPPED_SPEED_KMH
        and pd.notna(seconds_since_start)
        and seconds_since_start < STARTUP_GUARD_SECONDS
    ):
        return "STARTUP_WARMUP"

    if speed <= STOPPED_SPEED_KMH:
        return "IDLE"

    if pd.isna(accel):
        return "UNKNOWN_TRANSITION"

    if accel <= STRONG_DECEL_THRESHOLD_MPS2:
        return "BRAKING_DECEL"

    if accel <= DECEL_THRESHOLD_MPS2:
        return "COASTING_DECEL"

    if accel >= ACCEL_THRESHOLD_MPS2:
        return "ACCELERATING"

    return "CRUISING"
