import os
import numpy as np
import soundfile as sf
from .constants import FRAME_SEC

N_FFT = 512

# --- Tunable constants ---
_FRAME_SEC = 0.05           # Frame interval in seconds
_EPSILON = 1e-10            # Small value to avoid log(0) / div-by-zero
_ENV_AVG_WIN_SEC = 0.5      # Envelope sliding average window (seconds)
_ENV_MAX_WIN_SEC = 3.0      # Envelope sliding max window (seconds)
_ENV_DB_FLOOR = -18.0       # dB floor -> maps to 0
_ENV_DB_RANGE = 12.0        # dB range from floor to ceiling ([-18,-6] -> [0,1])


def load_audio(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ('.wav', '.flac', '.ogg'):
        y, sr = sf.read(file_path, always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if y.dtype.kind in 'iu':
            ii = np.iinfo(y.dtype)
            y = y.astype(np.float64) / max(abs(ii.min), ii.max)
    else:
        import audioread
        with audioread.audio_open(file_path) as af:
            sr = af.samplerate
            n_channels = af.channels
            chunks = []
            for buf in af:
                chunks.append(np.frombuffer(buf, dtype=np.int16))
            y = np.concatenate(chunks).astype(np.float64) / 32768.0
        if n_channels > 1:
            y = y.reshape(-1, n_channels).mean(axis=1)

    duration = len(y) / sr
    hop_length = int(FRAME_SEC * sr)
    return y, sr, duration, hop_length, N_FFT


def compute_rms(y, n_fft, hop_length):
    pad = n_fft // 2
    yp = np.pad(y, pad, mode='reflect')
    n_frames = int(np.ceil(len(y) / hop_length))
    rms = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        frame = yp[i * hop_length: i * hop_length + n_fft]
        rms[i] = np.sqrt(np.mean(frame * frame))
    return rms


def compute_stft_magnitude(y, n_fft, hop_length):
    pad = n_fft // 2
    yp = np.pad(y, pad, mode='reflect')
    n_frames = int(np.ceil(len(y) / hop_length))
    n_bins = n_fft // 2 + 1
    window = np.hanning(n_fft)
    stft = np.empty((n_bins, n_frames), dtype=np.float64)
    for i in range(n_frames):
        frame = yp[i * hop_length: i * hop_length + n_fft]
        stft[:, i] = np.abs(np.fft.rfft(frame * window))
    return stft


def compute_amplitude_envelope(rms_per_frame, frame_sec=_FRAME_SEC):
    peak = np.max(rms_per_frame)
    if peak <= 0:
        peak = _EPSILON

    dB = 20 * np.log10(rms_per_frame / peak + _EPSILON)

    avg_win = max(1, int(_ENV_AVG_WIN_SEC / frame_sec))
    if avg_win > 1:
        kernel = np.ones(avg_win) / avg_win
        dB = np.convolve(dB, kernel, mode='same')

    max_win = int(_ENV_MAX_WIN_SEC / frame_sec)
    half = max_win // 2
    n = len(dB)
    envelope = np.zeros(n)
    for i in range(n):
        s = max(0, i - half)
        e = min(n, i + half + 1)
        envelope[i] = float(np.max(dB[s:e]))

    return np.clip((envelope - _ENV_DB_FLOOR) / _ENV_DB_RANGE, 0.0, 1.0)
