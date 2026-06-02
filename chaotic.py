from pathlib import Path
from typing import Mapping, Iterable

import numpy as np
import pandas as pd
import pywt
import soundfile as sf
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_curve, auc


EPS = 1e-10


def _as_1d_float(y: np.ndarray | Iterable[float]):
    return np.asarray(y).reshape(-1).astype(float)


def normalize_signal(y: np.ndarray | Iterable[float], eps: float = EPS):
    y = _as_1d_float(y)
    y = y - y.mean()
    return y / (y.std() + eps)


# --------------------------------------
# Lyapunov exponent



def lyapunov_rosenstein(
    y: np.ndarray | Iterable[float],
    m: int = 5,
    tau: int = 1,
    max_iter: int = 20,
    fit_start: int = 1,
    fit_end: int = 10,):

    y = _as_1d_float(y)

    if len(y) < (m - 1) * tau + max_iter + 10:
        return np.nan

    n = len(y)
    phase_len = n - (m - 1) * tau

    x_emb = np.array([
        y[i:i + (m - 1) * tau + 1:tau]
        for i in range(phase_len)
    ])

    divergence_sum = np.zeros(max_iter)
    count = np.zeros(max_iter)
    theiler = m * tau

    for i in range(phase_len):
        dists = np.sqrt(((x_emb - x_emb[i]) ** 2).sum(axis=1))
        dists[max(0, i - theiler):min(phase_len, i + theiler + 1)] = np.inf

        j = np.argmin(dists)
        if not np.isfinite(dists[j]):
            continue

        for k in range(max_iter):
            if i + k < phase_len and j + k < phase_len:
                d_k = np.sqrt(((x_emb[i + k] - x_emb[j + k]) ** 2).sum())
                if d_k > 0:
                    divergence_sum[k] += np.log(d_k)
                    count[k] += 1

    mask = count > 0
    if mask.sum() <= 3:
        return np.nan

    divergence_sum[mask] /= count[mask]
    x = np.arange(max_iter)[mask]
    y_log = divergence_sum[mask]

    fit_end = min(fit_end, len(x))
    if fit_end - fit_start < 3:
        return np.nan

    slope = np.polyfit(x[fit_start:fit_end], y_log[fit_start:fit_end], 1)[0]
    return float(slope)


def wavelet_components(
    y: np.ndarray | Iterable[float],
    wavelet: str = "db4",
    level: int = 4,
    normalize: bool = True,):

    y = normalize_signal(y) if normalize else _as_1d_float(y)
    coeffs = pywt.wavedec(y, wavelet=wavelet, level=level)

    components: dict[str, np.ndarray] = {f"A{level}": coeffs[0]}

    for i, detail in enumerate(coeffs[1:], start=1):
        detail_level = level - i + 1
        components[f"D{detail_level}"] = detail

    return components


def wavelet_lyapunov_features(
    y: np.ndarray | Iterable[float],
    wavelet: str = "db4",
    level: int = 4,
    signal_len: int | None = 3000,
    m: int = 5,
    tau: int = 1,
    max_iter: int = 20,):

    y = _as_1d_float(y)
    y = y - y.mean()

    if signal_len is not None:
        y = y[:signal_len]

    comps = wavelet_components(y, wavelet=wavelet, level=level, normalize=True)

    return {
        f"lyap_{name}": lyapunov_rosenstein(
            comp,
            m=m,
            tau=tau,
            max_iter=max_iter,
        )
        for name, comp in comps.items()
    }


def compute_wavelet_lyapunov_for_folder(
    folder: str | Path,
    label_map: Mapping[str, str],
    wavelet: str = "db4",
    level: int = 4,
    signal_len: int | None = 3000,
    m: int = 5,
    tau: int = 1,
    max_iter: int = 20,):


    folder = Path(folder)
    rows = []

    for path in sorted(folder.glob("*.wav")):
        if path.name not in label_map:
            continue

        y, _ = sf.read(path)
        row = {
            "filename": path.name,
            "label": label_map[path.name],
        }
        row.update(
            wavelet_lyapunov_features(
                y,
                wavelet=wavelet,
                level=level,
                signal_len=signal_len,
                m=m,
                tau=tau,
                max_iter=max_iter,
            )
        )
        rows.append(row)

    return pd.DataFrame(rows).dropna()


# ------------------------------
# RQA



def embed_signal(
    y: np.ndarray | Iterable[float],
    m: int = 4,
    tau: int = 1,):

    y = _as_1d_float(y)
    n_vectors = len(y) - (m - 1) * tau

    if n_vectors <= 0:
        return None

    return np.array([
        y[i:i + m * tau:tau]
        for i in range(n_vectors)
    ])


def rqa_features(
    y: np.ndarray | Iterable[float],
    m: int = 4,
    tau: int = 1,
    eps_quantile: float = 0.1,
    min_diag: int = 2,
    normalize: bool = True,):
    
    y = normalize_signal(y) if normalize else _as_1d_float(y)
    x_emb = embed_signal(y, m=m, tau=tau)

    if x_emb is None or len(x_emb) < 20:
        return {"rr": np.nan, "det": np.nan, "lmean": np.nan}

    dist = np.sqrt(((x_emb[:, None, :] - x_emb[None, :, :]) ** 2).sum(axis=2))
    eps = np.quantile(dist, eps_quantile)
    recurrence_matrix = dist <= eps
    np.fill_diagonal(recurrence_matrix, False)

    rr = float(recurrence_matrix.mean())

    diag_lengths: list[int] = []

    for offset in range(-recurrence_matrix.shape[0] + 1, recurrence_matrix.shape[0]):
        diag = np.diagonal(recurrence_matrix, offset=offset)
        count = 0

        for val in diag:
            if val:
                count += 1
            elif count > 0:
                if count >= min_diag:
                    diag_lengths.append(count)
                count = 0

        if count >= min_diag:
            diag_lengths.append(count)

    if not diag_lengths:
        det = 0.0
        lmean = 0.0
    else:
        recurrent_points_in_diags = np.sum(diag_lengths)
        total_recurrent_points = recurrence_matrix.sum() + EPS
        det = float(recurrent_points_in_diags / total_recurrent_points)
        lmean = float(np.mean(diag_lengths))

    return {"rr": rr, "det": det, "lmean": lmean}


def rqa_wavelet_detail_features(
    y: np.ndarray | Iterable[float],
    detail_level: int = 1,
    wavelet: str = "db4",
    level: int = 4,
    signal_len: int | None = 2000,
    m: int = 4,
    tau: int = 1,
    eps_quantile: float = 0.1,
    min_diag: int = 2,):
 
    if detail_level < 1 or detail_level > level:
        raise ValueError("detail_level must satisfy 1 <= detail_level <= level")

    y = _as_1d_float(y)
    y = y - y.mean()

    if signal_len is not None:
        y = y[:signal_len]

    y = y / (y.std() + EPS)
    coeffs = pywt.wavedec(y, wavelet=wavelet, level=level)
    detail = coeffs[-detail_level]

    feats = rqa_features(
        detail,
        m=m,
        tau=tau,
        eps_quantile=eps_quantile,
        min_diag=min_diag,
        normalize=True,
    )

    suffix = f"D{detail_level}"
    return {
        f"rqa_rr_{suffix}": feats["rr"],
        f"rqa_det_{suffix}": feats["det"],
        f"rqa_lmean_{suffix}": feats["lmean"],
    }


def compute_rqa_wavelet_detail_for_folder(
    folder: str | Path,
    label_map: Mapping[str, str],
    detail_level: int = 1,
    wavelet: str = "db4",
    level: int = 4,
    signal_len: int | None = 2000,
    m: int = 4,
    tau: int = 1,
    eps_quantile: float = 0.1,
    min_diag: int = 2,):

  
    folder = Path(folder)
    rows = []

    for path in sorted(folder.glob("*.wav")):
        if path.name not in label_map:
            continue

        y, _ = sf.read(path)
        row = {
            "filename": path.name,
            "label": label_map[path.name],
        }
        row.update(
            rqa_wavelet_detail_features(
                y,
                detail_level=detail_level,
                wavelet=wavelet,
                level=level,
                signal_len=signal_len,
                m=m,
                tau=tau,
                eps_quantile=eps_quantile,
                min_diag=min_diag,
            )
        )
        rows.append(row)

    return pd.DataFrame(rows).dropna()


# -----------------------------------------------------------------------------
# Evaluation helper
# -----------------------------------------------------------------------------


def evaluate_binary_features(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    label_col: str = "label",
    positive_label: str = "path",
) -> pd.DataFrame:
    """
    Compute ROC AUC and Mann-Whitney p-value for each feature column.

    If AUC < 0.5, the feature direction is inverted and ``direction`` is set to -1.
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in {"filename", label_col}]

    y_true = (df[label_col] == positive_label).astype(int).values
    results = []

    for col in feature_cols:
        scores = df[col].values
        mask = ~np.isnan(scores)

        if mask.sum() == 0 or len(np.unique(y_true[mask])) < 2:
            results.append({
                "feature": col,
                "auc": np.nan,
                "pval": np.nan,
                "direction": np.nan,
                "mean_normal": np.nan,
                "mean_path": np.nan,
            })
            continue

        fpr, tpr, _ = roc_curve(y_true[mask], scores[mask])
        roc_auc = auc(fpr, tpr)
        direction = 1

        if roc_auc < 0.5:
            fpr, tpr, _ = roc_curve(y_true[mask], -scores[mask])
            roc_auc = auc(fpr, tpr)
            direction = -1

        normal_vals = df.loc[df[label_col] != positive_label, col].dropna()
        path_vals = df.loc[df[label_col] == positive_label, col].dropna()

        _, pval = mannwhitneyu(
            normal_vals,
            path_vals,
            alternative="two-sided",
        )

        results.append({
            "feature": col,
            "auc": float(roc_auc),
            "pval": float(pval),
            "direction": direction,
            "mean_normal": float(normal_vals.mean()),
            "mean_path": float(path_vals.mean()),
        })

    return (
        pd.DataFrame(results)
        .sort_values("auc", ascending=False)
        .reset_index(drop=True)
    )
