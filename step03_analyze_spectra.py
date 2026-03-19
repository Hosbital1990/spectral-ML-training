import pickle

import numpy as np

INPUT_FILE = "parsed_spectra.pkl"


def main():
    with open(INPUT_FILE, "rb") as f:
        samples = pickle.load(f)

    print("Total spectra loaded:", len(samples))

    starts = []
    ends = []
    lengths = []

    for sample in samples:
        x = sample["x"]

        starts.append(np.min(x))
        ends.append(np.max(x))
        lengths.append(len(x))

    starts = np.array(starts)
    ends = np.array(ends)
    lengths = np.array(lengths)

    print("\n--- Spectral Range Statistics ---")
    print("Smallest spectrum start:", starts.min())
    print("Largest spectrum start :", starts.max())
    print("Smallest spectrum end  :", ends.min())
    print("Largest spectrum end   :", ends.max())

    print("\n--- Spectrum Length Statistics ---")
    print("Minimum points:", lengths.min())
    print("Maximum points:", lengths.max())
    print("Average points:", lengths.mean())

    common_start = starts.max()
    common_end = ends.min()

    print("\n--- Strict Common Overlap ---")
    print("Common start:", common_start)
    print("Common end  :", common_end)
    print("Common width:", common_end - common_start)

    print("\n--- First 5 Spectra Ranges ---")
    for i in range(5):
        x = samples[i]["x"]
        label = samples[i]["label"]
        print(
            f"{i+1}. {label} | start={x.min():.2f} | end={x.max():.2f} | points={len(x)}"
        )


if __name__ == "__main__":
    main()