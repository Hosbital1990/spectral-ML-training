import os
import pickle
import random

import numpy as np

DATASET_PATH = "LR-Raman"
OUTPUT_FILE = "parsed_spectra.pkl"


def read_spectrum(file_path):
    label = None
    wavenumbers = []
    intensities = []

    with open(file_path, "r", encoding="latin-1", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if line.startswith("##NAMES="):
                label = line.split("=", 1)[1]

            if "," in line and not line.startswith("#"):
                try:
                    wavenumber, intensity = line.split(",")
                    wavenumbers.append(float(wavenumber))
                    intensities.append(float(intensity))
                except Exception:
                    pass

    return label, np.array(wavenumbers), np.array(intensities)


def collect_samples(dataset_path):
    samples = []

    for root, _, files in os.walk(dataset_path):
        for filename in files:
            if filename.lower().endswith(".txt"):
                path = os.path.join(root, filename)

                label, x, y = read_spectrum(path)

                if label is not None and len(x) > 1 and len(y) > 1:
                    samples.append({
                        "label": label,
                        "x": x,
                        "y": y,
                    })

    return samples


def main():
    samples = collect_samples(DATASET_PATH)

    print("Total parsed spectra:", len(samples))

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(samples, f)

    print("Dataset saved to:", OUTPUT_FILE)

    with open(OUTPUT_FILE, "rb") as f:
        loaded_samples = pickle.load(f)

    print("Reloaded dataset size:", len(loaded_samples))

    if len(loaded_samples) > 6000:
        sample = loaded_samples[6000]

        print("\nSample at index 6000")
        print("Label:", sample["label"])
        print("Number of points:", len(sample["x"]))
        print("First 5 wavenumbers:", sample["x"][:5])
        print("First 5 intensities:", sample["y"][:5])
    else:
        print("\nDataset has fewer than 6001 samples.")

    random_index = random.randint(0, len(loaded_samples) - 1)
    sample = loaded_samples[random_index]

    print("\nRandom sample index:", random_index)
    print("Label:", sample["label"])
    print("Number of points:", len(sample["x"]))
    print("First 5 wavenumbers:", sample["x"][:5])
    print("First 5 intensities:", sample["y"][:5])


if __name__ == "__main__":
    main()