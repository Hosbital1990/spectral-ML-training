# Raman Spectral Classification Pipeline

## Overview

This project builds a complete machine learning pipeline to classify minerals based on their Raman spectra.

The dataset consists of thousands of `.txt` files, each containing:

* Raman **wavenumber (x-axis)**
* Raman **intensity (y-axis)**
* Metadata including **mineral name (label)**

The goal is to transform raw spectral data into a **trainable machine learning dataset** and evaluate a baseline classification model.

---

## Environment Setup

Recommended Python version: **3.10+**

### 1) Create virtual environment

```bash
python3 -m venv .venv
```

### 2) Activate virtual environment

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

### 3) Install required packages

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Quick check

```bash
python --version
pip list
```

---

## Pipeline Summary

```
RAW TXT files
   ↓
Step 1: Dataset inspection
   ↓
Step 2: Parse spectra → (label, x, y)
   ↓
Step 2A: Save parsed dataset (parsed_spectra.pkl)
   ↓
Step 3: Analyze spectral ranges
   ↓
Step 4: Smart interpolation to common grid
   ↓
Step 5: Normalize spectra
   ↓
Step 6: Train classification model
```

---

## Run Steps (Order Matters)

Run scripts in this exact order:

```bash
python step01_dataset_review.py
python step02_parse_pickle_file.py
python step03_analyze_spectra.py
python step04_interpolate_dataset.py
python step05_normalize_dataset.py
python step06_train_model.py
```

### Required pickle files before each dependent step

* Before Step 3 or Step 4, create `parsed_spectra.pkl` by running Step 2.
* Before Step 5, create `ml_dataset.pkl` by running Step 4.
* Before Step 6, create `ml_dataset_normalized.pkl` by running Step 5.

If a required pickle file is missing, rerun the previous step that generates it.

---

## Step 1 - Dataset Inspection

## Objective

Verify dataset structure and ensure all spectrum files are accessible.

## Process

* Scan directory for `.txt` files
* Count number of spectra
* Inspect sample file paths

## Output

* Total number of files
* Confirmation: **1 file = 1 sample**

---

## Step 2 - Spectrum Parsing

## Objective

Extract usable data from raw files.

## Extracted Fields

From each file:

```
##NAMES=MineralName
wavenumber, intensity
```

## Resulting Structure

Each sample is converted to:

```python
{
    "label": "MineralName",
    "x": np.array([...]),   # wavenumbers
    "y": np.array([...])    # intensities
}
```

## Key Challenges

* Non-UTF8 encoding → solved using `latin-1`
* Irregular formatting → handled with safe parsing
* Missing/invalid values → skipped

---

## Step 2A - Dataset Caching

## Objective

Avoid re-parsing thousands of files repeatedly.

## Output

```
parsed_spectra.pkl
```

Contains all samples in memory-efficient serialized format.

## Benefits

* Faster subsequent processing
* Reproducibility
* Easier debugging

---

## Step 3 - Spectral Range Analysis

## Objective

Understand variability across spectra.

## Observations

* Different starting wavenumbers (some negative, some >2000)
* Different ending wavenumbers (up to ~6700)
* Different lengths (573 to 10,000+ points)

## Key Finding

There is **no strict common overlap** across all spectra.

## Implication

A fixed grid must be chosen **carefully**, not arbitrarily.

---

## Step 4 - Smart Interpolation (Core Step)

## Objective

Convert variable-length spectra into fixed-length vectors.

## Problem

Machine learning requires:

```
X.shape = (samples, features)
```

But raw spectra vary in:

* length
* range
* sampling resolution

## Solution

### 1. Data-driven grid selection

Instead of guessing, evaluate coverage:

```
Range [100–1200] → low retention
Range [250–1200] → high retention
Range [400–1200] → optimal
```

### 2. Final chosen grid

```
400 – 1200 cm⁻¹
step = 1
features = 801
```

### 3. Interpolation

Each spectrum is resampled:

```python
interp1d(x, y) → common_grid
```

### 4. Filtering

Spectra are skipped if:

* insufficient data
* invalid structure
* do not cover selected range

## Output

```
ml_dataset.pkl
```

Containing:

```python
X → (9815, 801)
y → labels
grid → wavenumber axis
```

## Result

* 9815 usable spectra
* only 14 skipped
* consistent feature space

---

## Step 5 - Normalization

## Objective

Remove scale differences between spectra.

## Problem

Intensity values vary widely:

```
Spectrum A max = 500
Spectrum B max = 50000
```

This biases the model.

## Solution

Per-spectrum normalization:

```python
spectrum = spectrum / max(spectrum)
```

## Output

```
ml_dataset_normalized.pkl
```

## Result

* Values scaled to ~[0, 1]
* Comparable spectra
* Improved model stability

---

## Step 6 - Model Training

## Objective

Train a classifier to predict mineral type.

## Challenges

* Very high number of classes (~2300)
* Many classes have few samples

## Solution

### Class filtering

Remove classes with fewer than N samples:

```python
MIN_SAMPLES_PER_CLASS = 3
```

### Result

* Samples: 7815
* Classes: 1274

### Train/Test split

* Stratified split
* Ensures class balance

## Model

```python
RandomForestClassifier
```

## Result

```
Accuracy ≈ 0.64
```

---

## Interpretation of Results

## Accuracy

64% accuracy across **1274 classes** is strong for:

* high class count
* spectral similarity between minerals

## Observations

* Better performance on well-represented classes
* Weak performance on rare classes
* Class imbalance impacts macro metrics

---

## Key Design Decisions

## Why interpolation?

* Converts variable-length signals to fixed-length vectors
* Preserves spectral shape
* Standard in spectroscopy ML

## Why 400–1200 cm⁻¹?

* Maximizes dataset retention
* Covers important Raman fingerprint region
* Avoids sparse low-frequency regions

## Why Random Forest?

* Handles nonlinear patterns
* Works well on tabular data
* No heavy tuning required

---

## Limitations

* Many rare classes remain underrepresented
* No baseline correction or smoothing applied
* No deep learning models used yet
* Spectral noise not explicitly handled

---

## Possible Improvements

## Data

* Increase `MIN_SAMPLES_PER_CLASS`
* Remove noisy spectra
* Apply baseline correction

## Features

* Peak extraction
* Spectral smoothing
* Derivative spectra

## Models

* XGBoost
* 1D CNN (for spectral data)
* Transformer models

## Evaluation

* Confusion matrix
* Top-k accuracy
* Class grouping

---

## Final Conclusion

This pipeline successfully transforms raw Raman spectroscopy data into a machine-learning-ready dataset and trains a baseline classifier.

Key achievements:

* Robust data parsing and cleaning
* Smart feature alignment via interpolation
* Effective normalization
* Working multi-class classifier with strong baseline performance

The dataset is confirmed to be **trainable and suitable for further research and model improvement**.

---
