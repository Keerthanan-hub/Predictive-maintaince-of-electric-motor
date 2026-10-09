"""Signal processing + feature extraction (pure functions, no I/O)."""
import numpy as np, pywt
from scipy import signal, stats
from scipy.signal import butter, sosfiltfilt, hilbert

def preprocess(x, fs, low=10.0, high=5000.0, order=4):
    """DC removal -> detrend -> Butterworth band-pass. Returns a NEW array."""
    x = np.asarray(x, float)
    x = signal.detrend(x - x.mean())
    sos = butter(int(order), [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)

def windows(x, size=2048, overlap=0.5):
    step = max(1, int(size * (1 - overlap)))
    n = (len(x) - size) // step + 1
    return np.stack([x[i * step:i * step + size] for i in range(n)]) if n > 0 else np.empty((0, size))

def spectrum(x, fs):
    n = len(x)
    return np.fft.rfftfreq(n, 1 / fs), np.abs(np.fft.rfft(x * np.hanning(n))) * 2 / n

def envelope_spectrum(x, fs):
    e = np.abs(hilbert(x)); return spectrum(e - e.mean(), fs)

def basic_stats(w):
    rms = np.sqrt(np.mean(w ** 2)); pk = np.max(np.abs(w)); mabs = np.mean(np.abs(w))
    return {"mean": w.mean(), "rms": rms, "variance": w.var(), "std": w.std(),
            "skewness": stats.skew(w), "kurtosis": stats.kurtosis(w, fisher=False),
            "peak": pk, "peak_to_peak": np.ptp(w), "crest_factor": pk / rms,
            "shape_factor": rms / mabs, "impulse_factor": pk / mabs,
            "clearance_factor": pk / np.mean(np.sqrt(np.abs(w))) ** 2}

def features(w, fs):
    """Time, frequency, envelope and wavelet features for one window."""
    f = basic_stats(w)
    fr, sp = spectrum(w, fs)
    f["dominant_freq"] = fr[np.argmax(sp[1:]) + 1]
    f["spectral_energy"] = np.sum(sp ** 2)
    f["spectral_centroid"] = np.sum(fr * sp) / np.sum(sp)
    ef, es = envelope_spectrum(w, fs)
    m = (ef > 5) & (ef < 1000)
    f["env_peak_freq"] = ef[m][np.argmax(es[m])]
    f["env_peak_amp"] = es[m].max()
    f["env_rms"] = np.sqrt(np.mean(es[m] ** 2))
    en = np.array([np.sum(c ** 2) for c in pywt.wavedec(w, "db4", level=4)])
    for i, v in enumerate(en / en.sum()):
        f[f"wavelet_e{i}"] = v
    return {k: float(v) for k, v in f.items()}

FEATURE_NAMES = list(features(np.random.randn(2048), 12000).keys())
