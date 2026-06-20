import numpy as np

# --- Tunable constants ---
_FREQ_BYTE_MAX = 255           # Maximum byte value for frequency data
_LOW_BAND_MIN_BINS = 4        # Minimum low-frequency bins count
_LOW_BAND_RATIO = 0.12        # Ratio of total bins considered low band
_ONSET_W_FLUX = 0.5           # Onset score weight for spectral flux
_ONSET_W_LOW_ENERGY = 0.2     # Onset score weight for low-band energy
_ONSET_W_LOW_RATIO = 0.2      # Onset score weight for low-band ratio
_ONSET_W_RMS_RISE = 1.4       # Onset score weight for RMS rise
_ONSET_W_ZCR = 0.08           # Onset score weight for ZCR deviation
_ONSET_ZCR_TARGET = 0.12      # Target ZCR for deviation penalty


def to_freq_data(frame_mag, n_freq_bins):
    peak = float(np.max(frame_mag)) if np.max(frame_mag) > 0 else 1.0
    return np.minimum(_FREQ_BYTE_MAX, np.floor(_FREQ_BYTE_MAX * frame_mag / peak)).astype(np.int32)


def low_band_features(freq_data, n_freq_bins):
    low_end = max(_LOW_BAND_MIN_BINS, int(n_freq_bins * _LOW_BAND_RATIO))
    low_sum = float(np.sum(freq_data[:low_end]))
    total_sum = float(np.sum(freq_data))
    low_energy = (low_sum / low_end) / _FREQ_BYTE_MAX if low_end > 0 else 0.0
    low_ratio = low_sum / total_sum if total_sum > 0 else 0.0
    return low_energy, low_ratio


def spectral_flux(freq_data, last_freq, n_freq_bins):
    if last_freq is None:
        return 0.0
    d = freq_data.astype(np.float64) - last_freq.astype(np.float64)
    return float(np.sum(np.maximum(d, 0))) / (n_freq_bins * _FREQ_BYTE_MAX)


def zero_crossing_rate(seg):
    if len(seg) <= 1:
        return 0.0
    zc = np.sum(
        ((seg[:-1] >= 0) & (seg[1:] < 0)) |
        ((seg[:-1] < 0) & (seg[1:] >= 0))
    )
    return zc / (len(seg) - 1)


def onset_score(flux, low_energy, low_ratio, rms_rise, zcr):
    return max(0.0, min(1.0,
        flux * _ONSET_W_FLUX +
        low_energy * _ONSET_W_LOW_ENERGY +
        low_ratio * _ONSET_W_LOW_RATIO +
        rms_rise * _ONSET_W_RMS_RISE -
        abs(zcr - _ONSET_ZCR_TARGET) * _ONSET_W_ZCR
    ))


def spectral_centroid(frame_mag, sr, n_fft):
    n_bins = len(frame_mag)
    freqs = np.linspace(0, sr / 2.0, n_bins)
    total = float(np.sum(frame_mag))
    if total < 1e-10:
        return 0.0
    return float(np.sum(freqs * frame_mag) / total)
