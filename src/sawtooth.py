# --- Tunable constants ---
_SERVO_MAX_SPEED = 500.0    # Max servo speed (pos/sec): 100pos / 0.2s


def clamp_for_servo(pos_from, pos_to, dt_ms):
    if dt_ms <= 0:
        return pos_from
    max_delta = _SERVO_MAX_SPEED * (dt_ms / 1000.0)
    delta = pos_to - pos_from
    if abs(delta) <= max_delta:
        return pos_to
    return pos_from + max_delta * (1.0 if delta > 0 else -1.0)
