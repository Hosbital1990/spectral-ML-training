import os

DATASET_PATH = "LR-Raman"


def collect_spectrum_files(dataset_path):
    spectrum_files = []
    for root, _, files in os.walk(dataset_path):
        for filename in files:
            if filename.lower().endswith(".txt"):
                spectrum_files.append(os.path.join(root, filename))
    return spectrum_files


def main():
    txt_files = collect_spectrum_files(DATASET_PATH)

    print("Total spectrum files found:", len(txt_files))

    print("\nFirst 5 files:")
    for file_path in txt_files[:5]:
        print(file_path)

    if txt_files:
        print("\nExample spectrum file:")
        print(txt_files[0])


if __name__ == "__main__":
    main()