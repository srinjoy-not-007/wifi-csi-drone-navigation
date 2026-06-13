#!/usr/bin/env python3
"""
Simple WiFi CSI baseline for room/location/activity classification.

This script is intentionally dataset-tolerant because public CSI datasets often
store data differently. It supports .npy, .npz, .mat, and .csv files, extracts
simple statistics from CSI windows, and trains a lightweight classifier.

Typical usage:
    python code/train_csi_baseline.py \
        --data_dir data/raw \
        --label_mode parent \
        --out_dir reports/run1 \
        --model rf

Demo mode, only to verify the pipeline:
    python code/train_csi_baseline.py --demo --out_dir reports/demo
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


SUPPORTED_EXTENSIONS = {".npy", ".npz", ".mat", ".csv"}
PREFERRED_KEYS = (
    "csi",
    "CSI",
    "csi_data",
    "csi_trace",
    "csi_matrix",
    "data",
    "amp",
    "amplitude",
    "features",
    "x",
    "X",
)


@dataclass
class Sample:
    features: np.ndarray
    label: str
    group: str
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple CSI baseline classifier.")
    parser.add_argument("--data_dir", type=Path, default=Path("data/raw"), help="Dataset root directory.")
    parser.add_argument("--out_dir", type=Path, default=Path("reports/run"), help="Where outputs are written.")
    parser.add_argument("--label_mode", choices=["parent", "filename", "csv"], default="parent")
    parser.add_argument("--label_csv", type=Path, default=None, help="CSV with columns: file,label. Required for --label_mode csv.")
    parser.add_argument("--window_size", type=int, default=64, help="Number of packets/time samples per window.")
    parser.add_argument("--stride", type=int, default=32, help="Stride between windows.")
    parser.add_argument("--max_files", type=int, default=None, help="Limit number of files for quick tests.")
    parser.add_argument("--time_axis", type=int, default=None, help="Override time axis. By default, uses heuristic.")
    parser.add_argument("--use_phase", action="store_true", help="Include phase-difference features when input is complex.")
    parser.add_argument("--model", choices=["rf", "logreg", "svm"], default="rf")
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--pca_components", type=int, default=64, help="PCA components for logreg/svm. Use 0 to disable.")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--demo", action="store_true", help="Run on generated synthetic CSI-like data; not a real result.")
    return parser.parse_args()


def load_label_csv(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        raise ValueError("--label_csv is required when --label_mode csv")
    df = pd.read_csv(path)
    required = {"file", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Label CSV must contain columns {required}, got {df.columns.tolist()}")
    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        mapping[str(row["file"])] = str(row["label"])
        mapping[Path(str(row["file"])).name] = str(row["label"])
    return mapping


def infer_label(path: Path, data_root: Path, mode: str, label_map: Dict[str, str]) -> str:
    if mode == "parent":
        return path.parent.name

    if mode == "csv":
        rel = str(path.relative_to(data_root))
        if rel in label_map:
            return label_map[rel]
        if path.name in label_map:
            return label_map[path.name]
        raise KeyError(f"No label for {rel} in label CSV.")

    # Filename mode: try common tokens, otherwise use prefix before first separator.
    stem = path.stem
    patterns = [
        r"(?:room|loc|location|class|label|zone|act|activity)[_-]?([A-Za-z0-9]+)",
        r"^([A-Za-z]+\d+)",
        r"^([A-Za-z0-9]+)[_-]",
    ]
    for pattern in patterns:
        match = re.search(pattern, stem, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return stem.split("_")[0].split("-")[0]


def discover_files(data_dir: Path, max_files: Optional[int]) -> List[Path]:
    files = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if max_files is not None:
        files = files[:max_files]
    return files


def _best_array_from_npz(npz_obj) -> np.ndarray:
    keys = list(npz_obj.keys())
    for key in PREFERRED_KEYS:
        if key in keys:
            return np.asarray(npz_obj[key])
    # Pick largest numeric array.
    arrays = []
    for key in keys:
        arr = np.asarray(npz_obj[key])
        if np.issubdtype(arr.dtype, np.number) or np.iscomplexobj(arr):
            arrays.append(arr)
    if not arrays:
        raise ValueError("No numeric arrays found in npz file.")
    return max(arrays, key=lambda a: a.size)


def _best_array_from_mat(mat_obj: dict) -> np.ndarray:
    clean = {k: v for k, v in mat_obj.items() if not k.startswith("__")}
    for key in PREFERRED_KEYS:
        if key in clean:
            return np.asarray(clean[key])
    candidates = []
    for _, value in clean.items():
        arr = np.asarray(value)
        if arr.size > 0 and (np.issubdtype(arr.dtype, np.number) or np.iscomplexobj(arr)):
            candidates.append(arr)
    if not candidates:
        raise ValueError("No numeric arrays found in mat file.")
    return max(candidates, key=lambda a: a.size)


def load_csi_file(path: Path) -> np.ndarray:
    ext = path.suffix.lower()
    if ext == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if ext == ".npz":
        with np.load(path, allow_pickle=False) as obj:
            return _best_array_from_npz(obj)
    if ext == ".mat":
        return _best_array_from_mat(loadmat(path))
    if ext == ".csv":
        return pd.read_csv(path, header=None).values
    raise ValueError(f"Unsupported file extension: {path}")


def choose_time_axis(arr: np.ndarray, override: Optional[int] = None) -> int:
    if override is not None:
        if override < 0:
            override += arr.ndim
        if override < 0 or override >= arr.ndim:
            raise ValueError(f"Invalid time axis {override} for shape {arr.shape}")
        return override
    if arr.ndim <= 1:
        return 0
    # CSI recordings usually have time/packet count as the largest axis.
    # If not, this is still a reasonable default and can be overridden.
    return int(np.argmax(arr.shape))


def to_time_channel_matrix(arr: np.ndarray, time_axis: Optional[int], use_phase: bool) -> np.ndarray:
    arr = np.asarray(arr)
    arr = np.squeeze(arr)

    if arr.ndim == 0:
        raise ValueError("Scalar array cannot be used as CSI.")

    if arr.ndim == 1:
        arr = arr[:, None]
    else:
        t_axis = choose_time_axis(arr, time_axis)
        arr = np.moveaxis(arr, t_axis, 0)
        arr = arr.reshape(arr.shape[0], -1)

    if np.iscomplexobj(arr):
        amp = np.abs(arr)
        if use_phase:
            phase = np.unwrap(np.angle(arr), axis=0)
            # Absolute phase often has random offsets; phase differences are more stable.
            phase_diff = np.diff(phase, axis=1, prepend=phase[:, :1])
            arr = np.concatenate([amp, phase_diff], axis=1)
        else:
            arr = amp
    else:
        arr = arr.astype(np.float64, copy=False)

    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def robust_window_features(window: np.ndarray) -> np.ndarray:
    """Return compact statistical features for a T x C window."""
    if window.ndim != 2:
        raise ValueError("window must be T x C")

    # Per-channel stats preserve subcarrier/antenna structure.
    mean = np.mean(window, axis=0)
    std = np.std(window, axis=0)
    q25 = np.percentile(window, 25, axis=0)
    q75 = np.percentile(window, 75, axis=0)
    iqr = q75 - q25

    diff = np.diff(window, axis=0)
    if diff.size == 0:
        diff_mean = np.zeros(window.shape[1])
        diff_std = np.zeros(window.shape[1])
    else:
        diff_mean = np.mean(np.abs(diff), axis=0)
        diff_std = np.std(diff, axis=0)

    # Global temporal features avoid exploding dimensionality too much.
    centered = window - mean[None, :]
    energy = np.mean(centered**2)
    abs_diff_energy = np.mean(np.abs(diff)) if diff.size else 0.0

    return np.concatenate([
        mean,
        std,
        iqr,
        diff_mean,
        diff_std,
        np.array([energy, abs_diff_energy], dtype=np.float64),
    ])


def iter_windows(matrix: np.ndarray, window_size: int, stride: int) -> Iterable[np.ndarray]:
    t = matrix.shape[0]
    if t < window_size:
        yield matrix
        return
    for start in range(0, t - window_size + 1, stride):
        yield matrix[start : start + window_size]


def samples_from_file(
    path: Path,
    label: str,
    args: argparse.Namespace,
) -> List[Sample]:
    arr = load_csi_file(path)
    matrix = to_time_channel_matrix(arr, args.time_axis, args.use_phase)
    samples: List[Sample] = []
    for window in iter_windows(matrix, args.window_size, args.stride):
        feat = robust_window_features(window)
        samples.append(Sample(features=feat, label=label, group=str(path), source=str(path)))
    return samples


def make_demo_samples(random_state: int) -> List[Sample]:
    """Synthetic CSI-like data only for checking the script; not a real experiment."""
    rng = np.random.default_rng(random_state)
    samples: List[Sample] = []
    n_classes = 4
    files_per_class = 8
    packets = 256
    subcarriers = 30
    antennas = 3
    for cls in range(n_classes):
        base_amp = rng.normal(loc=1.0 + cls * 0.2, scale=0.15, size=(subcarriers * antennas))
        freq = 0.01 + 0.004 * cls
        for fidx in range(files_per_class):
            t = np.arange(packets)[:, None]
            phase = rng.uniform(-np.pi, np.pi, size=(1, subcarriers * antennas))
            drift = 0.1 * np.sin(2 * np.pi * freq * t + phase)
            noise = rng.normal(scale=0.08, size=(packets, subcarriers * antennas))
            amp = base_amp[None, :] + drift + noise
            complex_csi = amp * np.exp(1j * (phase + 0.02 * t))
            for w in iter_windows(complex_csi, 64, 32):
                matrix = to_time_channel_matrix(w, None, use_phase=True)
                feat = robust_window_features(matrix)
                samples.append(Sample(feat, f"room_{cls}", f"demo_room_{cls}_file_{fidx}", "demo"))
    return samples


def load_samples(args: argparse.Namespace) -> List[Sample]:
    if args.demo:
        return make_demo_samples(args.random_state)

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    label_map = load_label_csv(args.label_csv) if args.label_mode == "csv" else {}
    files = discover_files(args.data_dir, args.max_files)
    if not files:
        raise FileNotFoundError(
            f"No CSI files found in {args.data_dir}. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    all_samples: List[Sample] = []
    failures: List[Tuple[str, str]] = []
    for path in files:
        try:
            label = infer_label(path, args.data_dir, args.label_mode, label_map)
            all_samples.extend(samples_from_file(path, label, args))
        except Exception as exc:  # Keep the run going and report bad files.
            failures.append((str(path), repr(exc)))

    if failures:
        print("Skipped files:")
        for fname, err in failures[:20]:
            print(f"  - {fname}: {err}")
        if len(failures) > 20:
            print(f"  ... plus {len(failures) - 20} more")

    if not all_samples:
        raise RuntimeError("No usable CSI samples could be extracted.")
    return all_samples


def build_model(args: argparse.Namespace, n_features: int) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]

    if args.model == "rf":
        clf = RandomForestClassifier(
            n_estimators=80,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=args.random_state,
            n_jobs=-1,
        )
        steps.append(("clf", clf))
        return Pipeline(steps)

    # Linear/SVM models benefit from scaling and optional PCA.
    steps.append(("scaler", StandardScaler()))
    if args.pca_components and args.pca_components > 0:
        n_components = min(args.pca_components, max(1, n_features - 1))
        steps.append(("pca", PCA(n_components=n_components, random_state=args.random_state)))

    if args.model == "logreg":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=args.random_state,
            n_jobs=-1,
        )
    elif args.model == "svm":
        clf = SVC(kernel="rbf", C=5.0, gamma="scale", class_weight="balanced")
    else:
        raise ValueError(f"Unknown model {args.model}")

    steps.append(("clf", clf))
    return Pipeline(steps)


def group_or_stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prefer group split by file to avoid window leakage; fallback if impossible."""
    unique_groups = np.unique(groups)
    if len(unique_groups) >= 4:
        try:
            splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            train_idx, test_idx = next(splitter.split(X, y, groups))
            if len(np.unique(y[train_idx])) == len(np.unique(y)) and len(np.unique(y[test_idx])) > 1:
                return train_idx, test_idx
        except Exception:
            pass

    # Fallback: stratified by sample. This can be optimistic if many windows come from same file.
    try:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        return next(splitter.split(X, y))
    except Exception:
        idx = np.arange(len(y))
        return train_test_split(idx, test_size=test_size, random_state=random_state)


def save_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: Path) -> None:
    if len(class_names) > 50:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(cm, aspect="auto")
        ax.set_title(f"Confusion matrix heatmap ({len(class_names)} classes)")
        ax.set_xlabel("Predicted class index")
        ax.set_ylabel("True class index")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args)
    X = np.vstack([s.features for s in samples])
    labels = np.array([s.label for s in samples])
    groups = np.array([s.group for s in samples])

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    all_label_ids = np.arange(len(encoder.classes_))

    train_idx, test_idx = group_or_stratified_split(X, y, groups, args.test_size, args.random_state)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = build_model(args, n_features=X.shape[1])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_test,
        y_pred,
        labels=all_label_ids,
        target_names=encoder.classes_,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=all_label_ids)

    metrics = {
        "demo_mode": bool(args.demo),
        "n_windows": int(len(samples)),
        "n_files_or_groups": int(len(np.unique(groups))),
        "n_features": int(X.shape[1]),
        "classes": encoder.classes_.tolist(),
        "model": args.model,
        "window_size": args.window_size,
        "stride": args.stride,
        "use_phase": bool(args.use_phase),
        "test_size": args.test_size,
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "classification_report": report,
        "split_note": "Group split by source file is attempted first to reduce window leakage; fallback is stratified sample split.",
    }

    with open(args.out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(args.out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Macro F1:  {f1_macro:.4f}\n")
        f.write(f"Classes:   {encoder.classes_.tolist()}\n")
        f.write(f"Windows:   {len(samples)}\n")
        f.write(f"Features:  {X.shape[1]}\n")
        f.write("\n")
        f.write(classification_report(y_test, y_pred, labels=all_label_ids, target_names=encoder.classes_, zero_division=0))

    save_confusion_matrix(cm, encoder.classes_, args.out_dir / "confusion_matrix.png")
    joblib.dump({"model": model, "label_encoder": encoder, "args": vars(args)}, args.out_dir / "model.joblib")

    print(f"Saved outputs to: {args.out_dir}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    if args.demo:
        print("NOTE: demo mode uses synthetic CSI-like data. Do not report this as a real dataset result.")


if __name__ == "__main__":
    main()
