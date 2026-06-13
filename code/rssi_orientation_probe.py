#!/usr/bin/env python3
"""
Optional live RSSI sanity check for Linux/Raspberry Pi/laptop.

This does NOT collect true CSI. It logs RSSI using `iw dev <iface> link`.
Use it only as a small physical sanity check: even simple WiFi strength changes
when the receiver is rotated/moved. CSI is richer and more sensitive, so a
moving drone will require careful pose/motion handling.

Example:
  python code/rssi_orientation_probe.py --iface wlan0 --out reports/rssi_probe.csv --seconds 90

During the run, rotate the laptop/RPi/receiver every 20-30 seconds:
  0 deg -> 90 deg -> 180 deg -> move/walk/vibrate.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import time
from pathlib import Path

RSSI_RE = re.compile(r"signal:\s*(-?\d+(?:\.\d+)?)\s*dBm")


def read_rssi_dbm(iface: str) -> float | None:
    try:
        out = subprocess.check_output(["iw", "dev", iface, "link"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        return None
    m = RSSI_RE.search(out)
    if not m:
        return None
    return float(m.group(1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iface", default="wlan0", help="WiFi interface, e.g. wlan0")
    parser.add_argument("--out", type=Path, default=Path("reports/rssi_probe.csv"))
    parser.add_argument("--seconds", type=float, default=90.0)
    parser.add_argument("--period", type=float, default=0.25)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    rows = []
    print("Logging RSSI. Rotate/move the receiver during the run. Press Ctrl+C to stop.")
    try:
        while time.time() - start < args.seconds:
            t = time.time() - start
            rssi = read_rssi_dbm(args.iface)
            rows.append((t, rssi if rssi is not None else ""))
            print(f"t={t:7.2f}s  rssi={rssi}")
            time.sleep(args.period)
    except KeyboardInterrupt:
        pass

    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["time_s", "rssi_dbm"])
        w.writerows(rows)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
