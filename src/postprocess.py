from .sawtooth import clamp_for_servo


def _sign(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


def offset_valleys_to_zero(actions):
    n = len(actions)
    if n <= 2:
        return actions

    at = [a["at"] for a in actions]
    pos = [a["pos"] for a in actions]

    is_valley = [False] * n
    for i in range(1, n - 1):
        prev_dir = _sign(pos[i] - pos[i - 1])
        next_dir = _sign(pos[i + 1] - pos[i])
        if prev_dir == -1 and next_dir in (1, 0):
            is_valley[i] = True
        elif prev_dir == 0 and next_dir == 1:
            j = i - 1
            while j > 0 and pos[j] == pos[j - 1]:
                j -= 1
            if j > 0 and pos[j] < pos[j - 1]:
                is_valley[i] = True

    current_offset = None
    for i in range(n):
        if is_valley[i]:
            current_offset = pos[i]
            pos[i] = 0
        elif current_offset is not None:
            pos[i] = max(0, pos[i] - current_offset)

    return [{"at": at[i], "pos": pos[i]} for i in range(n)]


def stretch_to_range(actions):
    n = len(actions)
    if n <= 1:
        return actions

    pos = [a["pos"] for a in actions]
    lo = min(pos)
    hi = max(pos)
    if hi <= lo:
        return actions

    rng = hi - lo
    scaled = [int(round((p - lo) / rng * 100.0)) for p in pos]
    return [{"at": actions[i]["at"], "pos": scaled[i]} for i in range(n)]


def enforce_servo_speed(actions):
    if len(actions) <= 1:
        return actions

    result = [actions[0]]
    for i in range(1, len(actions)):
        dt_ms = actions[i]["at"] - actions[i - 1]["at"]
        clamped_pos = clamp_for_servo(result[-1]["pos"], actions[i]["pos"], dt_ms)
        result.append({"at": actions[i]["at"], "pos": int(round(clamped_pos))})
    return result
