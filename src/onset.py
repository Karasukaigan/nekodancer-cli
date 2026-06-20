import numpy as np

_MIN_BPM = 65                 # Minimum detectable BPM
_MAX_BPM = 190                # Maximum detectable BPM
_MAX_HISTORY_MS = 9000        # OnsetTracker max history duration (ms)

# --- Tunable constants ---
_PULSE_DECAY = 0.92           # Pulse tracking decay factor
_PULSE_ATTACK = 0.08          # Pulse tracking attack factor
_AUTOCORR_MIN_PULSES = 120    # Minimum pulse count for autocorrelation
_AUTOCORR_MIN_SPAN_MS = 2800  # Minimum time span for autocorrelation (ms)
_AUTOCORR_MEAN_SUB = 0.6      # Mean subtraction factor for autocorrelation
_AUTOCORR_W_DBL = 0.35        # Double-frequency harmonic weight
_AUTOCORR_W_HALF = 0.2        # Half-frequency harmonic weight
_AUTOCORR_CONF_OFFSET = 0.1   # Confidence offset
_AUTOCORR_CONF_SCALE = 0.45   # Confidence scale factor
_BPM_CURVE_WINDOW_MS = 6000   # BPM sliding window duration (ms)
_BPM_CURVE_HOP_MS = 1500      # BPM sliding window hop (ms)
_BPM_CURVE_MIN_FRAMES = 120   # Minimum frames for BPM curve
_BPM_SMOOTHER_CONF_DECAY = 0.98       # Confidence decay when no valid BPM
_BPM_SMOOTHER_HIGH_ALPHA = 0.24       # High-confidence BPM smoothing alpha
_BPM_SMOOTHER_LOW_ALPHA = 0.1         # Low-confidence BPM smoothing alpha
_BPM_SMOOTHER_CONF_THRESH = 0.55      # Confidence threshold for high/low alpha
_BPM_SMOOTHER_CONF_MOMENTUM = 0.85    # Confidence smoothing momentum
_BPM_SMOOTHER_CONF_FACTOR = 0.15      # Confidence update factor
_FRAME_MS = 50.0                      # Frame interval (ms)


def _compute_pulses(onset_scores):
    pulses = []
    base = 0.0
    for onset in onset_scores:
        base = base * _PULSE_DECAY + onset * _PULSE_ATTACK
        pulse = max(0.0, onset - base * _PULSE_DECAY)
        pulses.append(pulse)
    return pulses


def _autocorrelate_bpm(pulses, timestamps):
    n = len(pulses)
    if n < _AUTOCORR_MIN_PULSES or len(timestamps) != n:
        return None, 0.0
    span_ms = timestamps[-1] - timestamps[0]
    if span_ms < _AUTOCORR_MIN_SPAN_MS:
        return None, 0.0
    avg_dt = (span_ms / (n - 1)) / 1000.0
    if avg_dt <= 0:
        return None, 0.0

    min_lag = max(2, int((60.0 / _MAX_BPM) / avg_dt))
    max_lag = min(n - 2, int((60.0 / _MIN_BPM) / avg_dt))
    if max_lag <= min_lag:
        return None, 0.0

    mean_val = float(np.mean(pulses))
    norm = np.maximum(0, np.array(pulses) - mean_val * _AUTOCORR_MEAN_SUB)
    energy = float(np.sum(norm ** 2))
    if energy < 1e-7:
        return None, 0.0

    best_lag = 0
    best_score = -1.0
    for lag in range(min_lag, max_lag + 1):
        cross = float(np.sum(norm[lag:] * norm[:-lag]))
        na = float(np.sum(norm[lag:] ** 2))
        nb = float(np.sum(norm[:-lag] ** 2))
        base = cross / np.sqrt(na * nb) if na > 0 and nb > 0 else 0.0

        half_lag = lag // 2
        half_score = 0.0
        if half_lag >= min_lag:
            c = float(np.sum(norm[half_lag:] * norm[:-half_lag]))
            ca = float(np.sum(norm[half_lag:] ** 2))
            cb = float(np.sum(norm[:-half_lag] ** 2))
            half_score = c / np.sqrt(ca * cb) if ca > 0 and cb > 0 else 0.0

        dbl_score = 0.0
        double_lag = lag * 2
        if double_lag <= max_lag:
            c = float(np.sum(norm[double_lag:] * norm[:-double_lag]))
            da = float(np.sum(norm[double_lag:] ** 2))
            db = float(np.sum(norm[:-double_lag] ** 2))
            dbl_score = c / np.sqrt(da * db) if da > 0 and db > 0 else 0.0

        score = base + dbl_score * _AUTOCORR_W_DBL + half_score * _AUTOCORR_W_HALF
        if score > best_score:
            best_score = score
            best_lag = lag

    if best_lag == 0 or best_score < 0:
        return None, 0.0

    bpm = 60.0 / (best_lag * avg_dt)
    while bpm < _MIN_BPM:
        bpm *= 2
    while bpm > _MAX_BPM:
        bpm /= 2

    confidence = max(0.0, min(1.0, (best_score - _AUTOCORR_CONF_OFFSET) / _AUTOCORR_CONF_SCALE))
    return bpm, confidence


def compute_bpm_curve(onset_scores, timestamps,
                      window_ms=_BPM_CURVE_WINDOW_MS,
                      hop_ms=_BPM_CURVE_HOP_MS):
    n = len(onset_scores)
    if n == 0:
        return np.full(0, 60.0)

    window_frames = max(_BPM_CURVE_MIN_FRAMES, int(round(window_ms / _FRAME_MS)))
    hop_frames = max(1, int(round(hop_ms / _FRAME_MS)))

    pulses = _compute_pulses(onset_scores)

    if n < window_frames:
        bpm, _ = _autocorrelate_bpm(pulses, timestamps)
        if bpm is None or bpm <= 0:
            bpm = 60.0
        return np.full(n, bpm)

    centers = []
    bpm_vals = []

    last_start = n - window_frames
    for start in range(0, last_start + 1, hop_frames):
        end = start + window_frames
        bpm, _ = _autocorrelate_bpm(pulses[start:end], timestamps[start:end])
        if bpm is not None and bpm > 0:
            bpm_vals.append(bpm)
            centers.append((timestamps[start] + timestamps[end - 1]) * 0.5)

    if not bpm_vals:
        return np.full(n, 60.0)

    return np.interp(timestamps, centers, bpm_vals)


def compute_global_bpm(onset_scores, timestamps):
    pulses = _compute_pulses(onset_scores)
    bpm, _ = _autocorrelate_bpm(pulses, timestamps)
    return bpm if (bpm is not None and bpm > 0) else 60.0


_BEAT_SNAP_TOLERANCE = 0.3         # ±fraction of beat period for snap window
_BEAT_SMOOTH_FRAMES = 5            # Onset smoothing window for beat detection
_BEAT_PEAK_THRESH_FACTOR = 0.5     # Peak threshold = mean + factor * std


def detect_beats(onset_scores, timestamps, bpm=None):
    n = len(onset_scores)
    if n < 4:
        return np.array(timestamps, dtype=np.float64)

    if bpm is None or bpm <= 0:
        bpm = compute_global_bpm(onset_scores, timestamps)

    beat_period_ms = 60000.0 / bpm
    frame_ms = _FRAME_MS

    scores = np.array(onset_scores, dtype=np.float64)
    k = _BEAT_SMOOTH_FRAMES
    if k > 1 and n > k:
        kernel = np.ones(k) / k
        smooth = np.convolve(scores, kernel, mode='same')
    else:
        smooth = scores.copy()

    ts = np.array(timestamps, dtype=np.float64)

    threshold = float(np.mean(smooth) + _BEAT_PEAK_THRESH_FACTOR * np.std(smooth))
    peaks = []
    for i in range(1, n - 1):
        if smooth[i] > smooth[i - 1] and smooth[i] > smooth[i + 1] and smooth[i] > threshold:
            peaks.append(i)

    if not peaks:
        count = max(1, int((ts[-1] - ts[0]) / beat_period_ms))
        return np.linspace(ts[0], ts[-1], count + 1)

    peak_scores = [smooth[p] for p in peaks]
    start_idx = peaks[int(np.argmax(peak_scores))]
    start_time = ts[start_idx]

    grid_before = []
    t = start_time - beat_period_ms
    while t >= ts[0]:
        grid_before.append(t)
        t -= beat_period_ms
    grid_before.reverse()

    grid_after = [start_time]
    t = start_time + beat_period_ms
    while t <= ts[-1]:
        grid_after.append(t)
        t += beat_period_ms

    grid = grid_before + grid_after

    tolerance = beat_period_ms * _BEAT_SNAP_TOLERANCE
    beats = []
    snapped = []
    for g in grid:
        best_i = -1
        best_score = -1.0
        for pi in peaks:
            pt = ts[pi]
            if abs(pt - g) <= tolerance and smooth[pi] > best_score:
                best_score = smooth[pi]
                best_i = pi
        if best_i >= 0:
            beats.append(ts[best_i])
            snapped.append(True)
        else:
            beats.append(g)
            snapped.append(False)

    if not any(snapped):
        count = max(1, int((ts[-1] - ts[0]) / beat_period_ms))
        return np.linspace(ts[0], ts[-1], count + 1)

    first_snap = next(i for i, s in enumerate(snapped) if s)
    last_snap = len(snapped) - 1 - next(i for i, s in enumerate(reversed(snapped)) if s)
    beats = beats[first_snap:last_snap + 1]

    if beats and beats[-1] < ts[-1]:
        t = beats[-1] + beat_period_ms
        while t < ts[-1]:
            beats.append(t)
            t += beat_period_ms

    beats = sorted(set(beats))

    min_gap = beat_period_ms * 0.5
    filtered = [beats[0]]
    for b in beats[1:]:
        if b - filtered[-1] >= min_gap:
            filtered.append(b)

    return np.array(filtered, dtype=np.float64)


_REGULARIZE_WINDOW = 8             # Sliding window size (beats) for interval smoothing


def regularize_beats(beats):
    n = len(beats)
    if n < 3:
        return beats

    intervals = np.diff(beats)
    win = min(_REGULARIZE_WINDOW, len(intervals))
    if win > 1:
        pad_left = (win - 1) // 2
        pad_right = win // 2
        padded = np.pad(intervals, (pad_left, pad_right), mode='edge')
        kernel = np.ones(win) / win
        smooth_intervals = np.convolve(padded, kernel, mode='valid')
    else:
        smooth_intervals = intervals

    result = np.empty(n)
    result[0] = beats[0]
    for i in range(len(smooth_intervals)):
        result[i + 1] = result[i] + smooth_intervals[i]

    return result


_RHYTHM_FFT_N_CANDIDATES = 4       # Number of rhythm frequency candidates
_RHYTHM_FFT_MIN_HZ = 0.3           # Minimum rhythm frequency for FFT search
_RHYTHM_FFT_MAX_HZ = 8.0           # Maximum rhythm frequency for FFT search
_RHYTHM_FFT_PAD_FACTOR = 4         # Zero-padding factor for FFT resolution


def extract_rhythm_frequencies(onset_scores, timestamps, n_candidates=_RHYTHM_FFT_N_CANDIDATES,
                               window_ms=_BPM_CURVE_WINDOW_MS,
                               hop_ms=_BPM_CURVE_HOP_MS):
    n = len(onset_scores)
    if n == 0:
        return np.full((0, n_candidates), 1.0)

    frame_rate = 1000.0 / _FRAME_MS
    window_frames = max(_BPM_CURVE_MIN_FRAMES, int(round(window_ms / _FRAME_MS)))
    hop_frames = max(1, int(round(hop_ms / _FRAME_MS)))

    scores = np.array(onset_scores, dtype=np.float64)
    kernel = np.ones(3) / 3.0
    scores_smooth = np.convolve(scores, kernel, mode='same')

    if n < window_frames:
        pulses = _compute_pulses(onset_scores)
        bpm, _ = _autocorrelate_bpm(pulses, timestamps)
        if bpm is None or bpm <= 0:
            bpm = 120.0
        base_hz = bpm / 60.0
        candidates = np.array([base_hz / (2 ** i) for i in range(n_candidates)])
        candidates = np.maximum(candidates, _RHYTHM_FFT_MIN_HZ)
        return np.tile(candidates, (n, 1))

    centers = []
    freq_candidates_list = []

    last_start = n - window_frames
    for start in range(0, last_start + 1, hop_frames):
        end = start + window_frames
        segment = scores_smooth[start:end].copy()
        segment -= np.mean(segment)

        win = np.hanning(len(segment))
        segment *= win

        fft_size = int(2 ** np.ceil(np.log2(len(segment) * _RHYTHM_FFT_PAD_FACTOR)))
        spectrum = np.abs(np.fft.rfft(segment, n=fft_size))
        freqs = np.fft.rfftfreq(fft_size, d=1.0 / frame_rate)

        mask = (freqs >= _RHYTHM_FFT_MIN_HZ) & (freqs <= _RHYTHM_FFT_MAX_HZ)
        masked_spec = np.where(mask, spectrum, 0.0)

        candidates = []
        for idx in range(1, len(masked_spec) - 1):
            if masked_spec[idx] > masked_spec[idx - 1] and masked_spec[idx] > masked_spec[idx + 1]:
                if masked_spec[idx] > 0:
                    candidates.append((masked_spec[idx], freqs[idx]))

        candidates.sort(key=lambda x: x[0], reverse=True)
        top_freqs = [c[1] for c in candidates[:n_candidates]]

        while len(top_freqs) < n_candidates:
            if top_freqs:
                top_freqs.append(top_freqs[0] / (len(top_freqs) + 1))
            else:
                top_freqs.append(1.0)

        top_freqs.sort(reverse=True)
        freq_candidates_list.append(top_freqs)
        centers.append((timestamps[start] + timestamps[end - 1]) * 0.5)

    if not freq_candidates_list:
        return np.full((n, n_candidates), 1.0)

    freq_arr = np.array(freq_candidates_list)
    result = np.empty((n, n_candidates))
    for c in range(n_candidates):
        result[:, c] = np.interp(timestamps, centers, freq_arr[:, c])

    return result


class OnsetTracker:
    def __init__(self, max_history_ms=_MAX_HISTORY_MS):
        self.env = []
        self.ts = []
        self.base = 0.0
        self.max_history_ms = max_history_ms

    def update(self, at_ms, onset):
        self.base = self.base * _PULSE_DECAY + onset * _PULSE_ATTACK
        pulse = max(0.0, onset - self.base * _PULSE_DECAY)
        self.env.append(pulse)
        self.ts.append(at_ms)
        while self.ts and self.ts[0] < at_ms - self.max_history_ms:
            self.ts.pop(0)
            self.env.pop(0)

    def estimate_bpm(self):
        return _autocorrelate_bpm(self.env, self.ts)


class BpmSmoother:
    def __init__(self):
        self.smooth_bpm = 0.0
        self.confidence = 0.0

    def update(self, bpm, confidence):
        if bpm is None or bpm <= 0:
            self.confidence *= _BPM_SMOOTHER_CONF_DECAY
            return
        if self.smooth_bpm <= 0:
            self.smooth_bpm = bpm
        alpha = _BPM_SMOOTHER_HIGH_ALPHA if confidence > _BPM_SMOOTHER_CONF_THRESH else _BPM_SMOOTHER_LOW_ALPHA
        self.smooth_bpm = self.smooth_bpm * (1 - alpha) + bpm * alpha
        self.confidence = self.confidence * _BPM_SMOOTHER_CONF_MOMENTUM + confidence * _BPM_SMOOTHER_CONF_FACTOR
