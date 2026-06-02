from pathlib import Path
import math
from itertools import permutations

import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks, windows, spectrogram as scipy_spectrogram
from scipy.stats import mannwhitneyu, entropy
from scipy.interpolate import interp1d

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.neighbors import NearestNeighbors
import random

import torch
import torchaudio
import librosa
import pywt

#CONSTANTS
SR          = 3000
FREQ_MAX    = 800
MIN_PROM_DB = 8.0
COMMON_FREQS = np.linspace(0, FREQ_MAX, 500)

#ENTROPY-COMPLEXITY
#-----------------------------------------

def shannon_entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def jensen_shannon(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)
    return shannon_entropy(m) - 0.5 * shannon_entropy(p) - 0.5 * shannon_entropy(q)


def permutation_distribution(x, m, tau):
    x = np.asarray(x, dtype=float)
    n = len(x)

    S = math.factorial(m)

    # сколько можно взять стартовых позиций, чтобы влезло i + (m-1)*tau
    K = n - (m - 1) * tau
    if K <= 0:
        raise ValueError("Too big tau or m ")

    perms = list(permutations(range(m)))
    perm_index = {perm: i for i, perm in enumerate(perms)}
    counts = np.zeros(S, dtype=float)

    for i in range(K):
        window = x[i : i + m * tau : tau]
        pattern = tuple(np.argsort(window, kind="mergesort"))
        counts[perm_index[pattern]] += 1.0

    return counts / counts.sum()

def entropy_complexity(x, m, tau):
    p = permutation_distribution(x, m, tau)
    S = len(p)

    pe = np.ones(S, dtype=float) / S

    H = shannon_entropy(p) / np.log(S)

    JS = jensen_shannon(p, pe)

    delta = np.zeros(S, dtype=float)
    delta[0] = 1.0
    JS_max = jensen_shannon(delta, pe)

    C = (JS * H) / JS_max
    return H, C

def compute_features_for_folder(folder: Path, label_map: dict,
                                m_values=(3,4), tau_values=(1,2,3,5,10)):
    folder = Path(folder)
    rows = []

    for p in sorted(folder.glob("*.wav")):
        if p.name not in label_map:
            continue

        y, sr = sf.read(p)
        y = np.asarray(y).reshape(-1).astype(float)

        lab = label_map[p.name]

        for m in m_values:
            for tau in tau_values:
                try:
                    H, C = entropy_complexity(y, m, tau)
                except Exception:
                    continue

                rows.append({
                    "filename": p.name,
                    "label": lab,
                    "m": m,
                    "tau": tau,
                    "H": float(H),
                    "C": float(C),
                    "sr": sr,
                    "n": len(y)
                })
    return pd.DataFrame(rows)


#KNN
#-----------------------------------------

def knn_label_agreement(df_feat, m, tau, k=5):
    g = df_feat[(df_feat["m"] == m) & (df_feat["tau"] == tau)].copy()
    g = g[g["label"].isin(["normal", "path"])].copy()
    if len(g) < k + 2:
        return np.nan

    X = g[["H", "C"]].values
    y = g["label"].values

    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(X)
    idx = nn.kneighbors(X, return_distance=False)[:, 1:]  # убрали саму точку

    return float((y[idx] == y[:, None]).mean())

def knn_agreement_perm_pvalue(df_feat, m, tau, k=5, n_perm=1000, seed=42):
    g = df_feat[(df_feat["m"] == m) & (df_feat["tau"] == tau)].copy()
    g = g[g["label"].isin(["normal", "path"])].copy()
    if len(g) < k + 2:
        return np.nan, np.nan

    X = g[["H", "C"]].values
    y = g["label"].values

    nn = NearestNeighbors(n_neighbors=k+1)
    nn.fit(X)
    idx = nn.kneighbors(X, return_distance=False)[:, 1:]

    def stat(labels):
        return (labels[idx] == labels[:, None]).mean()

    obs = stat(y)

    rng = np.random.default_rng(seed)
    perm = np.empty(n_perm)
    for t in range(n_perm):
        yp = y.copy()
        rng.shuffle(yp)
        perm[t] = stat(yp)

    p = (np.sum(perm >= obs) + 1) / (n_perm + 1)
    return float(obs), float(p)


def knn_perm_heatmaps(df_feat, m_values=(3,4), tau_values=range(1,11), k=5, n_perm=1000):
    agree = np.full((len(m_values), len(tau_values)), np.nan)
    pvals = np.full((len(m_values), len(tau_values)), np.nan)

    for i, m in enumerate(m_values):
        for j, tau in enumerate(tau_values):
            a, p = knn_agreement_perm_pvalue(df_feat, m, tau, k=k, n_perm=n_perm)
            agree[i, j] = a
            pvals[i, j] = p

    # agreement heatmap
    plt.figure(figsize=(12, 3.8))
    plt.imshow(agree, aspect="auto")
    plt.colorbar(label=f"Label agreement (k={k})")
    plt.xticks(np.arange(len(tau_values)), list(tau_values))
    plt.yticks(np.arange(len(m_values)), list(m_values))
    plt.xlabel("tau"); plt.ylabel("m")
    plt.title("kNN label agreement")
    for i in range(agree.shape[0]):
        for j in range(agree.shape[1]):
            if np.isfinite(agree[i, j]):
                plt.text(j, i, f"{agree[i,j]:.2f}", ha="center", va="center", fontsize=10)
    plt.savefig('KNN_method.png', dpi=300, bbox_inches='tight')
    plt.show()
    return agree, pvals

#COMPUTATION OF SPECTROGRAMS
#-----------------------------------------

def compute_spectrogram(y, sr, nperseg=512, noverlap=384, to_db=True):
    y = np.asarray(y).reshape(-1).astype(float)
    y = y - y.mean()
    f, t, Sxx = scipy_spectrogram(
        y, fs=sr, window="hann",
        nperseg=nperseg, noverlap=noverlap,
        scaling="density", mode="psd"
    )
    if to_db:
        Sxx = 10 * np.log10(Sxx + 1e-12)
    return f, t, Sxx

def compute_spectrogram_torchaudio(
    y, sr,
    n_fft=512,
    hop_length=128,
    win_length=None,
    center=False,
):
    """
    Torchaudio STFT -> спектрограмма мощности -> dB

    Возвращает:
      f (np.ndarray): частоты [Hz], размер (F,)
      t (np.ndarray): время [s], размер (T,)
      Sxx_db (np.ndarray): силы в dB, размер (F, T)
    """

    y = np.asarray(y).reshape(-1).astype(np.float32)
    y = y - float(y.mean())

    y_t = torch.from_numpy(y)

    if win_length is None:
        win_length = n_fft
    spec = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window_fn=torch.hann_window,
        power=2.0,     # power spectrogram
        center=center,
        pad=0,
        normalized=False
    )(y_t)  # (F, T)

    spec_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=None)(spec)  # (F, T)

    # axes
    f = torch.linspace(0.0, sr / 2.0, steps=spec_db.shape[0])
    t = torch.arange(spec_db.shape[1], dtype=torch.float32) * (hop_length / sr)

    return f.cpu().numpy(), t.cpu().numpy(), spec_db.cpu().numpy()

def detect_whistles_in_spectrogram(
    f, t, Sxx_db,
    fmin=0, fmax=1500,
    min_prom_db=MIN_PROM_DB,
    max_peaks_per_frame=3,
    min_freq_hz=80
):
    """
    - Для каждого временного окна: находятся пики по частотной оси с заметным пиком >= min_prom_db
    - Сохраняются лучшие N щначений по вспышкам
    - Отбрасывает низкие частоты
    Returns:
      whistle_points: (время, частота, сила)
      all_peak_freqs: список частот
      (f_b, S_b): ось частот и спектрограмма
    """
    band = (f >= fmin) & (f <= fmax)
    f_b = f[band]
    S_b = Sxx_db[band, :]

    whistle_points = []
    all_peak_freqs = []

    for j in range(S_b.shape[1]):
        col = S_b[:, j]

        peaks, props = find_peaks(col, prominence=min_prom_db)
        if len(peaks) == 0:
            continue

        prom = props["prominences"]
        order = np.argsort(prom)[::-1]
        peaks = peaks[order][:max_peaks_per_frame]
        prom = prom[order][:max_peaks_per_frame]

        for idx, p_idx in enumerate(peaks):
            freq = float(f_b[p_idx])
            if freq < min_freq_hz:
                continue
            strength = float(prom[idx])
            whistle_points.append((float(t[j]), freq, strength))
            all_peak_freqs.append(freq)

    return np.array(whistle_points), np.array(all_peak_freqs), (f_b, S_b)

def plot_spectrogram_with_whistles(f_b, t, S_b, whistle_points, title="", fmax_plot=800):
    plt.figure(figsize=(12, 5))
    plt.pcolormesh(t, f_b, S_b, shading="auto")
    plt.ylim(0, fmax_plot)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(title)
    plt.colorbar(label="Power (dB)")

    if whistle_points is not None and len(whistle_points) > 0:
        plt.scatter(whistle_points[:, 0], whistle_points[:, 1], s=12, marker="o")
    plt.show()

def frame_peak_freqs(f, Sxx_db, fmin=0, fmax=800, min_prom_db=MIN_PROM_DB, min_freq_hz=80, max_peaks_per_frame=3):
    band = (f >= fmin) & (f <= fmax)
    fb = f[band]
    Sb = Sxx_db[band, :]
    freqs_out = []

    for j in range(Sb.shape[1]):
        col = Sb[:, j]
        peaks, props = find_peaks(col, prominence=min_prom_db)
        if len(peaks) == 0:
            continue
        prom = props["prominences"]
        order = np.argsort(prom)[::-1]
        peaks = peaks[order][:max_peaks_per_frame]
        for p in peaks:
            fr = float(fb[p])
            if fr >= min_freq_hz:
                freqs_out.append(fr)
    return np.array(freqs_out, dtype=float)



def file_band_presence(p: Path,
                       band=(360, 390),
                       nperseg=512, noverlap=384,
                       min_prom_db=8.0, fmax=800,
                       min_freq_hz=80, max_peaks_per_frame=3):
    y, sr = sf.read(p)
    f, t, Sxx_db = compute_spectrogram(y, sr, nperseg=nperseg, noverlap=noverlap)

    # в каждом кадре времени ищем пики и проверяем в диапазоне ли
    band_lo, band_hi = band
    band_hits = 0
    total_frames = 0
    peak_freqs_all = []

    # делаем как frame_peak_freqs, но по кадрам, чтобы считать долю кадров
    band_mask = (f >= 0) & (f <= fmax)
    fb = f[band_mask]
    Sb = Sxx_db[band_mask, :]


    for j in range(Sb.shape[1]):
        col = Sb[:, j]
        peaks, props = find_peaks(col, prominence=min_prom_db)
        total_frames += 1
        if len(peaks) == 0:
            continue

        prom = props["prominences"]
        order = np.argsort(prom)[::-1]
        peaks = peaks[order][:max_peaks_per_frame]

        freqs_j = []
        for idx in peaks:
            fr = float(fb[idx])
            if fr >= min_freq_hz:
                freqs_j.append(fr)

        if len(freqs_j) == 0:
            continue

        peak_freqs_all.extend(freqs_j)
        # hit если хотя бы один пик кадра попал в диапазон
        if np.any((np.array(freqs_j) >= band_lo) & (np.array(freqs_j) <= band_hi)):
            band_hits += 1

    presence = band_hits / max(total_frames, 1)
    return presence, total_frames, np.array(peak_freqs_all, dtype=float)

def build_band_table(folder: Path, label_map, band=(360, 390),
                     nperseg=512, noverlap=384,
                     min_prom_db=8.0, fmax=800):
    rows = []
    wavs = sorted(folder.glob("*.wav"))

    for p in wavs:
        if p.name not in label_map:
            continue
        lab = label_map[p.name]
        presence, total_frames, peak_freqs = file_band_presence(
            p, band=band, nperseg=nperseg, noverlap=noverlap,
            min_prom_db=min_prom_db, fmax=fmax
        )
        rows.append({
            "filename": p.name,
            "label": lab,
            "frames": total_frames,
            "presence_360_390": float(presence),
            "n_peaks_total": int(len(peak_freqs))
        })

    return pd.DataFrame(rows)


def plot_band_dynamics_zoom(folder: Path, label_map, band=(360, 390),
                            fmin_plot=350, fmax_plot=400,
                            nperseg=512, noverlap=384,
                            min_prom_db=MIN_PROM_DB, n_files=6,
                            target_label="path",
                            mode="first"):
    all_files = [p for p in sorted(folder.glob("*.wav"))
                 if p.name in label_map
                 and label_map[p.name] == target_label]

    if mode == "first":
        files = all_files[:n_files]
    elif mode == "last":
        files = all_files[-n_files:]
    elif mode == "random":
        files = random.sample(all_files, min(n_files, len(all_files)))
    else:
        raise ValueError(f"Unknown mode: {mode!r}. Use 'first', 'last' or 'random'")


    band_lo, band_hi = band

    for p in files:
        fig, ax = plt.subplots(1, 1, figsize=(14, 3))

        y, sr = sf.read(p)
        f, t, Sxx_db = compute_spectrogram(y, sr, nperseg=nperseg, noverlap=noverlap)

        plot_mask = (f >= fmin_plot) & (f <= fmax_plot)
        ax.pcolormesh(t, f[plot_mask], Sxx_db[plot_mask, :],
                      shading="auto", cmap="magma")
        ax.set_ylim(fmin_plot, fmax_plot)

        hit_times, hit_freqs = [], []
        fb = f[plot_mask]
        Sb = Sxx_db[plot_mask, :]

        for j in range(Sb.shape[1]):
            col = Sb[:, j]
            peaks, props = find_peaks(col, prominence=min_prom_db)
            for pk in peaks:
                fr = fb[pk]
                if band_lo <= fr <= band_hi:
                    hit_times.append(t[j])
                    hit_freqs.append(fr)

        if hit_times:
            hit_times = np.array(hit_times)
            hit_freqs = np.array(hit_freqs)
            order = np.argsort(hit_times)
            ax.plot(hit_times[order], hit_freqs[order],
                    color="cyan", lw=1, alpha=0.5)
            ax.scatter(hit_times, hit_freqs,
                       color="cyan", s=20, alpha=0.9,
                       label=f"peaks {band_lo}–{band_hi} Hz (n={len(hit_times)})")

        ax.axhspan(band_lo, band_hi, alpha=0.15, color="cyan")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time (s)")
        ax.set_title(f"{p.name} [{target_label}]")
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.savefig(f"band_dynamics_{target_label}_{p.stem}_{band_lo}_{band_hi}.png",
                    dpi=150, bbox_inches="tight")
        plt.show()


def peak_hist_for_group(files, bins, label_map, target_label,
                        nperseg=512, noverlap=384,
                        min_prom_db=MIN_PROM_DB, fmin=0, fmax=800,
                        min_freq_hz=80):

    counts = np.zeros(len(bins) - 1, dtype=float)
    n_files_used = 0

    for p in files:
        if p.name not in label_map:
            continue
        if label_map[p.name] != target_label:
            continue

        y, sr = sf.read(p)
        f, t, Sxx_db = compute_spectrogram(y, sr, nperseg=nperseg, noverlap=noverlap)
        peak_freqs = frame_peak_freqs(f, Sxx_db, fmin=fmin, fmax=fmax,
                                      min_prom_db=min_prom_db,
                                      min_freq_hz=min_freq_hz)
        if peak_freqs.size == 0:
            continue

        h, _ = np.histogram(peak_freqs, bins=bins)
        counts += h
        n_files_used += 1

    return counts, n_files_used

#SPECTRAL FEATURES
#-----------------------------------------

def compute_spectrum(y, sr, window="hann"):
    y = np.asarray(y, dtype=float)
    y -= y.mean()
    win = windows.hann(len(y))
    Y = rfft(y * win)
    freqs = rfftfreq(len(y), 1 / sr)
    power = np.abs(Y) ** 2 / np.sum(win ** 2)
    power /= power.sum()
    return freqs, power

def interpolate_spectrum(freqs, power, common_freqs):
    f = interp1d(freqs, power, bounds_error=False, fill_value=0)
    return f(common_freqs)

def extract_spectral_features(freqs, power, freq_max=FREQ_MAX):
    mask = freqs <= freq_max
    f, p = freqs[mask], power[mask]
    p = p / p.sum()

    centroid = np.sum(f * p)
    cum = np.cumsum(p); cum /= cum[-1]
    rolloff85 = f[np.searchsorted(cum, 0.85)]
    rolloff50 = f[np.searchsorted(cum, 0.50)]
    dom_freq = f[np.argmax(p)]
    spread = np.sqrt(np.sum(((f - centroid) ** 2) * p))

    slope, _ = np.polyfit(f, p, 1)
    flatness = np.exp(np.mean(np.log(p + 1e-12))) / (np.mean(p) + 1e-12)

    skewness = np.sum(((f - centroid) ** 3) * p) / (spread ** 3 + 1e-12)
    kurtosis = np.sum(((f - centroid) ** 4) * p) / (spread ** 4 + 1e-12)



    spec_entropy = entropy(p + 1e-12, base=2)

    max_entropy = np.log2(len(p))
    spectral_entropy = spec_entropy / max_entropy


    bands = [(0, 25), (25, 50), (50, 100), (100, 175)]
    band_powers = []
    for fl, fh in bands:
        m = (f >= fl) & (f < fh)
        band_powers.append(np.sum(p[m]))

    return {
        "centroid": centroid,
        "rolloff_85": rolloff85,
        "rolloff_50": rolloff50,
        "dom_freq": dom_freq,
        "spread": spread,
        "slope": slope,
        "flatness": flatness,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "spectral_entropy": spectral_entropy,
        "band_0_25": band_powers[0],
        "band_25_50": band_powers[1],
        "band_50_100": band_powers[2],
        "band_100_175": band_powers[3],
    }


def compute_spectral_features_for_folder(folder: Path, label_map: dict,
                                         freq_max=FREQ_MAX):

    rows = []
    for p in sorted(folder.glob("*.wav")):
        fname = p.name
        if fname not in label_map:
            continue
        y, sr = sf.read(p)
        y = np.asarray(y).reshape(-1)
        freqs, power = compute_spectrum(y, sr)
        features = extract_spectral_features(freqs, power, freq_max=freq_max)
        features["filename"] = fname
        features["label"] = label_map[fname]
        features["sr"] = sr
        features["duration_sec"] = len(y) / sr
        rows.append(features)
    return pd.DataFrame(rows)

#WAVELET ENTROPY
#-----------------------------------------

def compute_wavelet_entropy(y, sr, wavelet='db4', level=5):

    y = np.asarray(y, dtype=float)
    y -= y.mean()
    y /= (y.std() + 1e-10)


    coeffs = pywt.wavedec(y, wavelet=wavelet, level=level)

    features = {}
    for i, c in enumerate(coeffs):

        power = c ** 2
        power_norm = power / (power.sum() + 1e-12)


        ent = -np.sum(power_norm * np.log(power_norm + 1e-12))

        max_ent = np.log(len(power_norm))
        features[f"wavelet_entropy_level_{i}"] = ent / (max_ent + 1e-12)

    return features


def compute_wavelet_features_for_folder(folder: Path, label_map: dict,
                                        wavelet="db4", level=5):

    rows = []
    for p in sorted(folder.glob("*.wav")):
        if p.name not in label_map:
            continue
        y, sr = sf.read(p)
        y = np.asarray(y).reshape(-1)
        feats = compute_wavelet_entropy(y, sr, wavelet=wavelet, level=level)
        feats["filename"] = p.name
        feats["label"] = label_map[p.name]
        rows.append(feats)
    return pd.DataFrame(rows)

#     MFCC
#-----------------------------------------

def compute_mfcc_features_for_folder(folder: Path, label_map: dict,
                                     sr=SR, n_mfcc=13):

    rows = []
    for p in sorted(folder.glob("*.wav")):
        if p.name not in label_map:
            continue
        y, _ = librosa.load(str(p), sr=sr, mono=True)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        row = {"filename": p.name}
        for i in range(n_mfcc):
            row[f"mfcc_mean_{i}"] = mfcc[i].mean()
            row[f"mfcc_std_{i}"] = mfcc[i].std()
        rows.append(row)
    return pd.DataFrame(rows)

#PULSE IRREGULARITY
def estimate_pulse_irregularity(filepath, band=(360, 390),
                                 nperseg=512, noverlap=384,
                                 min_prom_db=8.0):
    y, sr = sf.read(filepath)
    f, t, Sxx_db = compute_spectrogram(y, sr, nperseg=nperseg, noverlap=noverlap)
    band_lo, band_hi = band
    fb_mask = (f >= 0) & (f <= 800)
    fb = f[fb_mask]
    Sb = Sxx_db[fb_mask, :]
    dt = t[1] - t[0] if len(t) > 1 else 0

    presence = []
    for j in range(Sb.shape[1]):
        col = Sb[:, j]
        peaks, props = find_peaks(col, prominence=min_prom_db)
        hit = any(band_lo <= fb[pk] <= band_hi for pk in peaks)
        presence.append(float(hit))
    presence = np.array(presence)

    transitions_on  = np.where(np.diff(presence) > 0)[0]
    transitions_off = np.where(np.diff(presence) < 0)[0]

    if len(transitions_on) >= 2:
        intervals = np.diff(transitions_on) * dt
        cv = intervals.std() / (intervals.mean() + 1e-10)
        mean_interval = intervals.mean()
    else:
        cv = np.nan
        mean_interval = np.nan

    on_lengths = []
    if len(transitions_on) > 0 and len(transitions_off) > 0:
        for on in transitions_on:
            offs_after = transitions_off[transitions_off > on]
            if len(offs_after) > 0:
                on_lengths.append((offs_after[0] - on) * dt)
    on_mean = np.mean(on_lengths) if on_lengths else np.nan
    on_std  = np.std(on_lengths)  if on_lengths else np.nan

    return {
        "cv_interval":     cv,
        "mean_interval":   mean_interval,
        "on_duration_mean": on_mean,
        "on_duration_std":  on_std,
        "presence_mean":   presence.mean(),
        "presence_std":    presence.std()
    }