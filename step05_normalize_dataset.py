import pickle

import numpy as np

INPUT_FILE = "ml_dataset.pkl"
OUTPUT_FILE = "ml_dataset_normalized.pkl"

NORMALIZATION_MODE = "max"

def normalize_dataset(X, y):
    X_norm = np.zeros_like(X, dtype=np.float64)
    skipped = []

    for i in range(len(X)):
        spectrum = X[i].astype(np.float64)

        if NORMALIZATION_MODE == "max":
            denom = np.max(np.abs(spectrum))
            if denom == 0:
                skipped.append((i, y[i], "zero max"))
                X_norm[i] = spectrum
            else:
                X_norm[i] = spectrum / denom

        elif NORMALIZATION_MODE == "l2":
            denom = np.linalg.norm(spectrum)
            if denom == 0:
                skipped.append((i, y[i], "zero l2 norm"))
                X_norm[i] = spectrum
            else:
                X_norm[i] = spectrum / denom

        elif NORMALIZATION_MODE == "zscore":
            mean_val = np.mean(spectrum)
            std_val = np.std(spectrum)
            if std_val == 0:
                skipped.append((i, y[i], "zero std"))
                X_norm[i] = spectrum - mean_val
            else:
                X_norm[i] = (spectrum - mean_val) / std_val

        else:
            raise ValueError(f"Unknown NORMALIZATION_MODE: {NORMALIZATION_MODE}")

    return X_norm, skipped


def main():
    with open(INPUT_FILE, "rb") as f:
        data = pickle.load(f)

    X = data["X"]
    y = data["y"]
    grid = data["grid"]

    print("Loaded dataset:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Grid length:", len(grid))

    X_norm, skipped = normalize_dataset(X, y)

    print("\n--- NORMALIZATION REPORT ---")
    print("Normalization mode:", NORMALIZATION_MODE)
    print("Normalized X shape:", X_norm.shape)
    print("Skipped/problematic spectra:", len(skipped))

    if len(X_norm) > 0:
        print("\nSample 0 stats before normalization:")
        print("min:", np.min(X[0]))
        print("max:", np.max(X[0]))
        print("mean:", np.mean(X[0]))

        print("\nSample 0 stats after normalization:")
        print("min:", np.min(X_norm[0]))
        print("max:", np.max(X_norm[0]))
        print("mean:", np.mean(X_norm[0]))

    print("\nFirst 5 problematic samples:")
    for item in skipped[:5]:
        print(item)

    normalized_data = {
        "X": X_norm,
        "y": y,
        "grid": grid,
        "start": data.get("start"),
        "end": data.get("end"),
        "step": data.get("step"),
        "normalization_mode": NORMALIZATION_MODE,
    }

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(normalized_data, f)

    print("\nSaved normalized dataset to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()