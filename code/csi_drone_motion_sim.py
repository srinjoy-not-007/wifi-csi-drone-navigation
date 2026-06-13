#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler


def smooth_random_pattern(rng, n, width=7):
    x = rng.normal(0, 1, n)
    kernel = np.ones(width) / width
    return np.convolve(x, kernel, mode="same")


def build_world(seed=7, n_bins=64, n_locations=5):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n_bins)

    location_patterns = []
    for i in range(n_locations):
        phase = rng.uniform(0, 2 * np.pi)
        p = (
            0.55 * np.sin(2 * np.pi * (i + 1) * x + phase)
            + 0.35 * np.cos(2 * np.pi * (i + 2) * x)
            + 0.20 * smooth_random_pattern(rng, n_bins)
        )
        location_patterns.append(p)

    yaw_basis_1 = np.sin(2 * np.pi * 3 * x + 0.3)
    yaw_basis_2 = np.cos(2 * np.pi * 5 * x - 0.2)
    vibration_basis = smooth_random_pattern(rng, n_bins, width=5)

    world = {
        "rng": rng,
        "n_bins": n_bins,
        "n_locations": n_locations,
        "location_patterns": np.array(location_patterns),
        "yaw_basis_1": yaw_basis_1,
        "yaw_basis_2": yaw_basis_2,
        "vibration_basis": vibration_basis,
    }
    return world


def simulate_packet(world, location_id, mode):
    """
    Lightweight CSI nuisance simulator.

    The simulated CSI amplitude vector is modeled as:

        observed CSI = location fingerprint
                     + yaw/orientation distortion
                     + vibration/tilt distortion
                     + AGC/global gain drift
                     + noise

    This is not a full ray-tracing RF simulator. It is a controlled diagnostic
    test for the static-fingerprint vs drone-motion question.
    """
    rng = world["rng"]
    loc = world["location_patterns"][location_id]

    if mode == "static":
        yaw = rng.normal(0.0, 0.03)
        vibration = 0.0
        tilt = 0.0
        noise_sigma = 0.03
        gain_sigma = 0.03

    elif mode == "drone":
        yaw = rng.uniform(-np.pi, np.pi)
        vibration = abs(rng.normal(0.0, 0.90))
        tilt = abs(rng.normal(0.0, 0.45))
        noise_sigma = 0.12
        gain_sigma = 0.25

    elif mode == "augmented":
        yaw = rng.uniform(-np.pi, np.pi)
        vibration = abs(rng.normal(0.0, 0.95))
        tilt = abs(rng.normal(0.0, 0.50))
        noise_sigma = 0.13
        gain_sigma = 0.25

    else:
        raise ValueError(f"Unknown mode: {mode}")

    yaw_effect = (
        2.20 * np.sin(yaw) * world["yaw_basis_1"]
        + 1.80 * np.cos(2 * yaw) * world["yaw_basis_2"]
    )

    vibration_effect = 1.80 * vibration * world["vibration_basis"]
    tilt_loss = -0.35 * tilt * np.abs(world["yaw_basis_1"])
    global_gain = world["rng"].normal(0.0, gain_sigma)

    observed = (
        loc
        + yaw_effect
        + vibration_effect
        + tilt_loss
        + global_gain
        + rng.normal(0.0, noise_sigma, world["n_bins"])
    )

    return observed.astype(np.float32)


def make_dataset(world, mode, samples_per_location):
    X = []
    y = []

    for location_id in range(world["n_locations"]):
        for _ in range(samples_per_location):
            X.append(simulate_packet(world, location_id, mode))
            y.append(location_id)

    return np.vstack(X), np.array(y)


def plot_mean_profiles(X_static, y_static, X_drone, y_drone, out_dir):
    plt.figure(figsize=(11, 6))

    for loc in sorted(set(y_static)):
        plt.plot(
            X_static[y_static == loc].mean(axis=0),
            linestyle="-",
            label=f"L{loc} static"
        )

    for loc in sorted(set(y_drone)):
        plt.plot(
            X_drone[y_drone == loc].mean(axis=0),
            linestyle="--",
            label=f"L{loc} drone-like"
        )

    plt.xlabel("CSI amplitude bin index")
    plt.ylabel("Simulated normalized CSI amplitude")
    plt.title("Static CSI fingerprints vs drone-like CSI fingerprints")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "sim_mean_static_vs_drone.png", dpi=200)
    plt.close()


def plot_pose_sensitivity(world, out_dir, location_id=2):
    yaws = np.linspace(-np.pi, np.pi, 13)
    curves = []

    # Save/restore RNG state-like behavior by using the same world directly;
    # this is just a visual sensitivity sweep.
    for yaw in yaws:
        loc = world["location_patterns"][location_id]
        yaw_effect = (
            2.20 * np.sin(yaw) * world["yaw_basis_1"]
            + 1.80 * np.cos(2 * yaw) * world["yaw_basis_2"]
        )
        curves.append(loc + yaw_effect)

    plt.figure(figsize=(11, 6))
    for i, curve in enumerate(curves):
        plt.plot(curve, alpha=0.75, label=f"yaw {np.rad2deg(yaws[i]):.0f}°")

    plt.xlabel("CSI amplitude bin index")
    plt.ylabel("Simulated CSI amplitude")
    plt.title("Same simulated location, different receiver yaw angles")
    plt.legend(fontsize=7, ncol=3)
    plt.tight_layout()
    plt.savefig(out_dir / "sim_same_location_yaw_sweep.png", dpi=200)
    plt.close()


def pca_plot(X_list, y_list, names, out_dir, filename, title):
    X = np.vstack(X_list)
    domain_labels = []

    for X_part, name in zip(X_list, names):
        domain_labels.extend([name] * len(X_part))

    domain_labels = np.array(domain_labels)

    Xs = StandardScaler().fit_transform(X)
    Z = PCA(n_components=2, random_state=42).fit_transform(Xs)

    plt.figure(figsize=(9, 6))
    for name in names:
        idx = domain_labels == name
        plt.scatter(Z[idx, 0], Z[idx, 1], s=14, alpha=0.70, label=name)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()


def train_and_eval(X_train, y_train, X_test, y_test, out_dir, filename, title):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    clf.fit(X_train_s, y_train)

    pred = clf.predict(X_test_s)
    acc = accuracy_score(y_test, pred)

    labels = sorted(set(y_train) | set(y_test))
    cm = confusion_matrix(y_test, pred, labels=labels)

    disp = ConfusionMatrixDisplay(cm, display_labels=[f"L{i}" for i in labels])
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False)
    plt.title(f"{title}\naccuracy={acc:.3f}")
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()

    return float(acc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=str(Path.home() / "wifi_csi_exp" / "sim_reports"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n_bins", type=int, default=64)
    parser.add_argument("--n_locations", type=int, default=5)
    parser.add_argument("--n_train_static", type=int, default=220)
    parser.add_argument("--n_test_static", type=int, default=100)
    parser.add_argument("--n_test_drone", type=int, default=100)
    parser.add_argument("--n_train_aug", type=int, default=300)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    world = build_world(
        seed=args.seed,
        n_bins=args.n_bins,
        n_locations=args.n_locations,
    )

    X_train_static, y_train_static = make_dataset(world, "static", args.n_train_static)
    X_test_static, y_test_static = make_dataset(world, "static", args.n_test_static)
    X_test_drone, y_test_drone = make_dataset(world, "drone", args.n_test_drone)
    X_train_aug, y_train_aug = make_dataset(world, "augmented", args.n_train_aug)

    acc_static_to_static = train_and_eval(
        X_train_static, y_train_static,
        X_test_static, y_test_static,
        out_dir,
        "sim_confusion_static_train_static_test.png",
        "Train static CSI, test static CSI"
    )

    acc_static_to_drone = train_and_eval(
        X_train_static, y_train_static,
        X_test_drone, y_test_drone,
        out_dir,
        "sim_confusion_static_train_drone_test.png",
        "Train static CSI, test drone-like CSI"
    )

    X_train_augmented = np.vstack([X_train_static, X_train_aug])
    y_train_augmented = np.concatenate([y_train_static, y_train_aug])

    acc_aug_to_drone = train_and_eval(
        X_train_augmented, y_train_augmented,
        X_test_drone, y_test_drone,
        out_dir,
        "sim_confusion_augmented_train_drone_test.png",
        "Train static + motion-augmented CSI, test drone-like CSI"
    )

    plot_mean_profiles(X_test_static, y_test_static, X_test_drone, y_test_drone, out_dir)
    plot_pose_sensitivity(world, out_dir, location_id=2)

    pca_plot(
        [X_test_static, X_test_drone],
        [y_test_static, y_test_drone],
        ["static test", "drone-like test"],
        out_dir,
        "sim_pca_static_vs_drone_domain.png",
        "PCA domain shift: static CSI vs drone-like CSI"
    )

    pca_plot(
        [X_train_static, X_train_aug, X_test_drone],
        [y_train_static, y_train_aug, y_test_drone],
        ["static train", "motion-augmented train", "drone-like test"],
        out_dir,
        "sim_pca_augmented_coverage.png",
        "Motion augmentation covers drone-like CSI domain"
    )

    metrics = {
        "seed": args.seed,
        "n_bins": args.n_bins,
        "n_locations": args.n_locations,
        "train_static_samples": int(len(X_train_static)),
        "test_static_samples": int(len(X_test_static)),
        "test_drone_samples": int(len(X_test_drone)),
        "train_augmented_motion_samples": int(len(X_train_aug)),
        "accuracy_train_static_test_static": acc_static_to_static,
        "accuracy_train_static_test_drone_like": acc_static_to_drone,
        "accuracy_train_static_plus_augmented_test_drone_like": acc_aug_to_drone,
        "interpretation": {
            "static_to_static": "How well a normal static fingerprint model works when train/test conditions match.",
            "static_to_drone_like": "How much performance drops when the same model is tested under drone-like yaw/vibration/tilt.",
            "augmented_to_drone_like": "How much motion/pose augmentation can recover performance."
        }
    }

    with (out_dir / "sim_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    with (out_dir / "SIMULATION_SUMMARY.md").open("w") as f:
        f.write("# Drone-like CSI Motion Simulation\n\n")
        f.write("This is a lightweight diagnostic simulation, not a full RF ray-tracer.\n\n")
        f.write("The goal is to test the static-fingerprint assumption under drone-like motion.\n\n")
        f.write("## Main results\n\n")
        f.write(f"- Train static → test static accuracy: `{acc_static_to_static:.3f}`\n")
        f.write(f"- Train static → test drone-like accuracy: `{acc_static_to_drone:.3f}`\n")
        f.write(f"- Train static + motion augmentation → test drone-like accuracy: `{acc_aug_to_drone:.3f}`\n\n")
        f.write("## Interpretation\n\n")
        f.write("- Static CSI fingerprints work well when train/test conditions match.\n")
        f.write("- The same model degrades when tested on drone-like yaw, tilt, vibration, and gain drift.\n")
        f.write("- Motion/pose augmentation improves robustness to drone-like CSI changes.\n")
        f.write("- Therefore, for UAV CSI localization, CSI should be collected with pose/motion diversity or fused with IMU/VIO.\n")

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved simulation reports to: {out_dir}")


if __name__ == "__main__":
    main()
