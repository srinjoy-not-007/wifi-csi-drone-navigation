#!/usr/bin/env python3

import argparse
import ast
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_csi_file(path: Path, max_packets=None):
    """
    Parse ESP32 CSI_DATA CSV lines.

    Each useful line looks like:
    CSI_DATA,...,"[83,-80,4,0,...]"

    We extract the final CSI vector and convert alternating I/Q values
    into magnitude-like amplitude features.
    """
    rows = []

    with path.open("r", errors="ignore") as f:
        reader = csv.reader(f)

        for row in reader:
            if not row:
                continue
            if row[0] != "CSI_DATA":
                continue

            try:
                csi_raw = ast.literal_eval(row[-1])
            except Exception:
                continue

            x = np.asarray(csi_raw, dtype=np.float32)

            # Make length even for I/Q pairing.
            if len(x) < 8:
                continue
            if len(x) % 2 == 1:
                x = x[:-1]

            # ESP32 CSI array is an interleaved signed integer sequence.
            # For this assignment, exact RF calibration is less important than
            # consistently extracting comparable packet-level features.
            i = x[0::2]
            q = x[1::2]
            amp = np.sqrt(i * i + q * q)

            # Remove extreme first few metadata-like / guard-looking bins only if desired.
            # Here we keep all bins, but normalize per packet to reduce gain effects.
            amp = np.nan_to_num(amp)

            if np.std(amp) > 1e-6:
                amp_norm = (amp - np.mean(amp)) / (np.std(amp) + 1e-6)
            else:
                amp_norm = amp - np.mean(amp)

            rows.append(amp_norm)

            if max_packets is not None and len(rows) >= max_packets:
                break

    if not rows:
        raise RuntimeError(f"No CSI_DATA rows parsed from {path}")

    # Some files may have slightly different CSI lengths. Pad/crop to common later.
    return rows


def load_dataset(raw_dir: Path, max_packets_per_file=None):
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise RuntimeError(f"No CSV files found in {raw_dir}")

    all_rows = []
    labels = []
    file_counts = {}

    parsed_by_file = {}
    min_len = None

    for path in files:
        label = path.stem
        rows = parse_csi_file(path, max_packets=max_packets_per_file)
        parsed_by_file[label] = rows
        file_counts[label] = len(rows)

        for r in rows:
            min_len = len(r) if min_len is None else min(min_len, len(r))

    for label, rows in parsed_by_file.items():
        for r in rows:
            all_rows.append(r[:min_len])
            labels.append(label)

    X = np.vstack(all_rows)
    y = np.asarray(labels)

    return X, y, file_counts


def make_window_features(X, y, window_size=32, stride=16):
    """
    Convert packet-level amplitude vectors into window-level features.
    Each window feature = mean, std, min, max over time for every subcarrier bin.
    """
    Xw = []
    yw = []

    for label in sorted(set(y)):
        idx = np.where(y == label)[0]
        X_label = X[idx]

        for start in range(0, len(X_label) - window_size + 1, stride):
            w = X_label[start:start + window_size]

            feat = np.concatenate([
                np.mean(w, axis=0),
                np.std(w, axis=0),
                np.min(w, axis=0),
                np.max(w, axis=0),
            ])

            Xw.append(feat)
            yw.append(label)

    return np.vstack(Xw), np.asarray(yw)


def plot_mean_amplitude(X, y, out_dir: Path):
    plt.figure(figsize=(11, 6))

    for label in sorted(set(y)):
        idx = np.where(y == label)[0]
        mean_amp = X[idx].mean(axis=0)
        plt.plot(mean_amp, label=label)

    plt.xlabel("CSI amplitude bin index")
    plt.ylabel("Per-packet normalized amplitude")
    plt.title("Mean CSI amplitude profile by condition")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "mean_amplitude_conditions.png", dpi=200)
    plt.close()


def plot_pca(Xw, yw, out_dir: Path, name="pca_all_conditions.png", title="PCA of CSI window features"):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xw)

    pca = PCA(n_components=2, random_state=42)
    Z = pca.fit_transform(Xs)

    plt.figure(figsize=(8, 6))

    for label in sorted(set(yw)):
        idx = np.where(yw == label)[0]
        plt.scatter(Z[idx, 0], Z[idx, 1], s=15, alpha=0.75, label=label)

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% var)")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / name, dpi=200)
    plt.close()

    return pca.explained_variance_ratio_.tolist()


def plot_pair_pca(Xw, yw, labels_to_use, out_dir: Path, name, title):
    mask = np.isin(yw, labels_to_use)
    if mask.sum() < 4:
        return None

    return plot_pca(Xw[mask], yw[mask], out_dir, name=name, title=title)


def train_condition_classifier(Xw, yw, out_dir: Path):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xw)

    X_train, X_test, y_train, y_test = train_test_split(
        Xs,
        yw,
        test_size=0.30,
        random_state=42,
        stratify=yw,
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    labels_sorted = sorted(set(yw))

    cm = confusion_matrix(y_test, pred, labels=labels_sorted)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title(f"Condition classification confusion matrix\naccuracy={acc:.3f}")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix_conditions.png", dpi=200)
    plt.close()

    report = classification_report(y_test, pred, labels=labels_sorted)

    with (out_dir / "classification_report.txt").open("w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write(report)

    return acc, report


def domain_shift_tests(Xw, yw, out_dir: Path):
    """
    Do simple targeted tests:
    1. Train on yaw0 static at A/B, test on yaw90 static at A/B.
    2. If vibration exists, train on static B_yaw0 and compare vibration as a separate condition.
    """
    results = {}

    # Yaw shift test: train location classifier on yaw0, test location classifier on yaw90.
    needed = {"A_yaw0_static", "B_yaw0_static", "A_yaw90_static", "B_yaw90_static"}
    if needed.issubset(set(yw)):
        train_mask = np.isin(yw, ["A_yaw0_static", "B_yaw0_static"])
        test_mask = np.isin(yw, ["A_yaw90_static", "B_yaw90_static"])

        X_train = Xw[train_mask]
        X_test = Xw[test_mask]

        # Map condition labels to location labels.
        y_train = np.array(["A" if "A_" in s else "B" for s in yw[train_mask]])
        y_test = np.array(["A" if "A_" in s else "B" for s in yw[test_mask]])

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1)
        clf.fit(X_train_s, y_train)
        pred = clf.predict(X_test_s)

        acc = accuracy_score(y_test, pred)
        results["train_yaw0_test_yaw90_location_accuracy"] = float(acc)

        labels = ["A", "B"]
        cm = confusion_matrix(y_test, pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, colorbar=False)
        plt.title(f"Yaw-shift location test\nTrain yaw0, test yaw90, acc={acc:.3f}")
        plt.tight_layout()
        plt.savefig(out_dir / "confusion_train_yaw0_test_yaw90.png", dpi=200)
        plt.close()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default=str(Path.home() / "wifi_csi_exp" / "raw"))
    parser.add_argument("--out_dir", type=str, default=str(Path.home() / "wifi_csi_exp" / "reports"))
    parser.add_argument("--max_packets_per_file", type=int, default=None)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y, counts = load_dataset(raw_dir, max_packets_per_file=args.max_packets_per_file)
    Xw, yw = make_window_features(X, y, window_size=args.window_size, stride=args.stride)

    plot_mean_amplitude(X, y, out_dir)
    explained = plot_pca(Xw, yw, out_dir)

    plot_pair_pca(
        Xw, yw,
        ["A_yaw0_static", "A_yaw90_static"],
        out_dir,
        "pca_same_location_yaw_A.png",
        "Same location A: yaw 0 vs yaw 90"
    )

    plot_pair_pca(
        Xw, yw,
        ["B_yaw0_static", "B_yaw90_static"],
        out_dir,
        "pca_same_location_yaw_B.png",
        "Same location B: yaw 0 vs yaw 90"
    )

    plot_pair_pca(
        Xw, yw,
        ["A_yaw0_static", "B_yaw0_static"],
        out_dir,
        "pca_location_change_yaw0.png",
        "Different locations: A yaw0 vs B yaw0"
    )

    plot_pair_pca(
        Xw, yw,
        ["B_yaw90_static", "B_yaw90_vibration"],
        out_dir,
        "pca_static_vs_vibration_B_yaw90.png",
        "Same location B, yaw 90: static vs vibration"
    )

    plot_pair_pca(
        Xw, yw,
        ["A_yaw0_static", "A_yaw0_vibration"],
        out_dir,
        "pca_static_vs_vibration_A.png",
        "Same location A: static vs vibration"
    )

    acc, report = train_condition_classifier(Xw, yw, out_dir)
    shift_results = domain_shift_tests(Xw, yw, out_dir)

    summary = {
        "raw_dir": str(raw_dir),
        "out_dir": str(out_dir),
        "packet_counts": counts,
        "packet_feature_shape": list(X.shape),
        "window_feature_shape": list(Xw.shape),
        "window_size": args.window_size,
        "stride": args.stride,
        "pca_explained_variance_ratio": explained,
        "condition_classifier_accuracy": float(acc),
        **shift_results,
    }

    with (out_dir / "metrics.json").open("w") as f:
        json.dump(summary, f, indent=2)

    with (out_dir / "summary.txt").open("w") as f:
        f.write("ESP32 CSI analysis summary\n")
        f.write("==========================\n\n")
        f.write(json.dumps(summary, indent=2))
        f.write("\n\nInterpretation notes:\n")
        f.write("- PCA separation between A_yaw0 and B_yaw0 suggests location affects CSI.\n")
        f.write("- PCA separation between A_yaw0 and A_yaw90 or B_yaw0 and B_yaw90 suggests receiver orientation affects CSI even at the same location.\n")
        f.write("- PCA separation between B_yaw90_static and B_yaw90_vibration suggests drone-like body motion changes CSI distribution even at same location/yaw.\n")
        f.write("- These are sanity checks, not a calibrated RF localization benchmark.\n")

    print(json.dumps(summary, indent=2))
    print(f"\nSaved plots and reports to: {out_dir}")


if __name__ == "__main__":
    main()
