# Drone-like CSI Motion Simulation

This is a lightweight diagnostic simulation, not a full RF ray-tracer.

The goal is to test the static-fingerprint assumption under drone-like motion.

## Main results

- Train static → test static accuracy: `1.000`
- Train static → test drone-like accuracy: `0.392`
- Train static + motion augmentation → test drone-like accuracy: `0.996`

## Interpretation

- Static CSI fingerprints work well when train/test conditions match.
- The same model degrades when tested on drone-like yaw, tilt, vibration, and gain drift.
- Motion/pose augmentation improves robustness to drone-like CSI changes.
- Therefore, for UAV CSI localization, CSI should be collected with pose/motion diversity or fused with IMU/VIO.
