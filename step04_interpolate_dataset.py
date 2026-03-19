import pickle

import numpy as np
from scipy.interpolate import interp1d

INPUT_FILE = "parsed_spectra.pkl"
OUTPUT_FILE = "ml_dataset.pkl"

GRID_END = 1200
GRID_STEP = 1

CANDIDATE_STARTS = [50, 100, 150, 200, 250, 300, 350, 400]

def choose_best_start(samples):
    print("\n--- Range Coverage Analysis ---")
    coverage_results = []

    for start in CANDIDATE_STARTS:
        kept = 0

        for sample in samples:
            x = sample["x"]
            y = sample["y"]

            if len(x) < 2 or len(y) < 2 or len(x) != len(y):
                continue

            xmin = x.min()
            xmax = x.max()

            if xmin <= start and xmax >= GRID_END:
                kept += 1

        coverage_results.append((start, kept))
        print(f"Range [{start}, {GRID_END}] -> kept={kept}")

    best_start = max(coverage_results, key=lambda item: item[1])[0]
    print("\nSelected best start:", best_start)
    return best_start


def interpolate_samples(samples, best_start):
    common_grid = np.arange(best_start, GRID_END + GRID_STEP, GRID_STEP)
    print("Grid size:", len(common_grid))

    features = []
    labels = []
    skipped = []

    for idx, sample in enumerate(samples):
        x = sample["x"]
        y_vals = sample["y"]
        label = sample["label"]

        if len(x) < 2 or len(y_vals) < 2 or len(x) != len(y_vals):
            skipped.append((idx, label, "length mismatch"))
            continue

        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y_vals[sort_idx]

        x_unique, unique_idx = np.unique(x_sorted, return_index=True)
        y_unique = y_sorted[unique_idx]

        if len(x_unique) < 2:
            skipped.append((idx, label, "not enough unique points"))
            continue

        xmin = x_unique.min()
        xmax = x_unique.max()
        if xmin > best_start or xmax < GRID_END:
            skipped.append((idx, label, "does not cover range"))
            continue

        interp_fn = interp1d(
            x_unique,
            y_unique,
            kind="linear",
            bounds_error=False,
            fill_value=0,
        )
        spectrum = interp_fn(common_grid)

        features.append(spectrum)
        labels.append(label)

    X = np.array(features)
    y = np.array(labels)
    return X, y, common_grid, skipped


def main():
    with open(INPUT_FILE, "rb") as f:
        samples = pickle.load(f)

    print("Loaded spectra:", len(samples))

    best_start = choose_best_start(samples)
    X, y, common_grid, skipped = interpolate_samples(samples, best_start)

    print("\n--- FINAL REPORT ---")
    print("Used range:", best_start, "to", GRID_END)
    print("Interpolated spectra:", len(X))
    print("Skipped spectra:", len(skipped))

    if len(X) > 0:
        print("Feature length:", X.shape[1])
        print("X shape:", X.shape)
        print("y shape:", y.shape)

    print("\nFirst 5 skipped samples:")
    for item in skipped[:5]:
        print(item)

    ml_dataset = {
        "X": X,
        "y": y,
        "grid": common_grid,
        "start": best_start,
        "end": GRID_END,
        "step": GRID_STEP,
    }

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(ml_dataset, f)

    print("\nSaved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()