# Public CSI Dataset Baseline: 250-Location Subset

This experiment uses a public CSI indoor-localization dataset. The full dataset was downloaded locally. After excluding real/imaginary-part-only folders, the coordinate-only set contained 688 `.mat` files. For a runtime-practical baseline, 250 coordinate/location classes were randomly sampled.

## Configuration

- Dataset subset: 250 coordinate/location classes
- Window size: 32
- Stride: 16
- Model: Random Forest
- Number of trees: 80
- Split: window-based train/test split
- Feature dimension: 452
- Number of windows: 15,260

## Results

- Accuracy: 0.9924
- Macro F1: 0.7827

## Interpretation

The high accuracy shows that CSI features contain strong location-specific information. The macro F1 is lower because several sampled classes had very few test windows, and some classes had zero or one sample in the test split. This experiment is therefore a practical CSI fingerprinting sanity check, not a strict cross-session or cross-environment deployment benchmark.
