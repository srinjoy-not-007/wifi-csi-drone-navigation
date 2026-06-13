#!/usr/bin/env python3
"""Quickly inspect a CSI dataset folder before training."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

SUPPORTED = {".npy", ".npz", ".mat", ".csv"}


def shape_of(path: Path):
    try:
        if path.suffix == ".npy":
            arr = np.load(path, allow_pickle=False)
            return arr.shape, str(arr.dtype)
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as obj:
                return {k: (obj[k].shape, str(obj[k].dtype)) for k in obj.keys()}, "npz"
        if path.suffix == ".mat":
            obj = loadmat(path)
            return {k: (np.asarray(v).shape, str(np.asarray(v).dtype)) for k, v in obj.items() if not k.startswith("__")}, "mat"
        if path.suffix == ".csv":
            df = pd.read_csv(path, header=None, nrows=5)
            return ("csv preview", df.shape), "csv"
    except Exception as exc:
        return "ERROR", repr(exc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--max_files", type=int, default=20)
    args = parser.parse_args()

    files = sorted(p for p in args.data_dir.rglob("*") if p.suffix.lower() in SUPPORTED)
    print(f"Found {len(files)} supported files under {args.data_dir}")
    for path in files[: args.max_files]:
        shape, dtype = shape_of(path)
        print(f"\n{path}")
        print(f"  shape: {shape}")
        print(f"  dtype: {dtype}")


if __name__ == "__main__":
    main()
