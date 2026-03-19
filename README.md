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
Step 6: Train & evaluate models (6 options available)
   ↓
Step 7: Compare all models (optional)
```

---

## Run Steps (Order Matters)

### Data preparation (required, run in order):

```bash
python3 step01_dataset_review.py
python3 step02_parse_pickle_file.py
python3 step03_analyze_spectra.py
python3 step04_interpolate_dataset.py
python3 step05_normalize_dataset.py
```

### Step 6 - Model Training (choose one or run all):

Pick any single model to train:

```bash
python3 step06_train_random_forest.py      # Default baseline
python3 step06_train_extra_trees.py        # Randomized ensemble variant
python3 step06_train_linear_svm.py         # Linear support vector machine
python3 step06_train_logistic_regression.py # Linear classifier
python3 step06_train_knn.py                # K-nearest neighbors
python3 step06_train_cnn_1d.py             # Convolutional neural network (requires TensorFlow)
```

### Step 7 - Compare All Models (optional):

Run all 6 models sequentially and print a ranked comparison table:

```bash
python3 step07_compare_models.py
```

This creates a `model_run_logs/` directory containing the full output of each model.

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

Train one or more classifiers to predict mineral type.

## Challenges

* Very high number of classes (~2300)
* Many classes have few samples
* Need to find best-performing model architecture

## Solution

### Class filtering (applied by all models)

Remove classes with fewer than N samples:

```python
MIN_SAMPLES_PER_CLASS = 3
```

### Result after filtering

* Samples: 7815
* Classes: 1274

### Train/Test split

* Stratified train/test split
* Ensures class balance

## Available Models

All models use the same data preparation pipeline and evaluation metrics.

### 1. Random Forest (default baseline)

**File:** `step06_train_random_forest.py`

* 100 decision trees
* Handles nonlinear patterns well
* Robust to high-dimensional data
* Expected accuracy: ~0.64

### 2. Extra Trees (extremely randomized trees)

**File:** `step06_train_extra_trees.py`

* 300 randomized trees
* Faster training than Random Forest
* Often comparable or better accuracy

### 3. Linear SVM (support vector machine)

**File:** `step06_train_linear_svm.py`

* Linear kernel for high-dimensional data
* Good for spectral classification
* Requires more training time

### 4. Logistic Regression

**File:** `step06_train_logistic_regression.py`

* Linear classifier with multinomial support
* Fast training and prediction
* Good baseline for linear separability

### 5. KNN (K-Nearest Neighbors)

**File:** `step06_train_knn.py`

* k=5 neighbors with distance weighting
* Lazy learner (no explicit training)
* Memory-intensive prediction

### 6. 1D CNN (convolutional neural network)

**File:** `step06_train_cnn_1d.py`

* Input: 1D spectral sequences with 2 conv layers
* Deep learning approach for spectral features
* Requires: TensorFlow/Keras
* Slower training but potentially higher accuracy on complex patterns

## Model Comparison

Run all 6 models and rank them:

```bash
python3 step07_compare_models.py
```

This outputs:

* Table with accuracy, runtime, and status for each model
* Per-model full logs in `model_run_logs/`
* Best-performing model highlighted

---

## Step 7 - Model Comparison

## Objective

Compare all 6 model implementations on the same dataset and identify best performer.

## How It Works

1. Sequentially runs each of the 6 Step 6 model scripts
2. Extracts accuracy, runtime, and error status from each
3. Stores full output logs in `model_run_logs/`
4. Displays ranked comparison table in terminal

## Output

A table showing:

* Model rank by accuracy
* Model name and status (ok/failed)
* Achieved accuracy
* Training + evaluation runtime
* Log file location

## Interpretation of Results

### Baseline Accuracy

~64% accuracy across **1274 classes** is strong for:

* high class count (>1000)
* spectral similarity between minerals
* limited samples per class

### Observations

* Tree-based models (Random Forest, Extra Trees) typically perform well on tabular data
* Linear models provide interpretable but potentially lower accuracy
* Neural networks (1D CNN) may excel with sufficient training data and tuning
* KNN is memory-intensive but can capture local patterns
* Model performance varies based on class representation

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

## Why multiple models?

* No single model dominates all classification problems
* Tree-based models handle nonlinear patterns well
* Linear models provide interpretability
* Deep learning scales with data quantity
* Systematic comparison identifies optimal architecture for this spectral dataset

---

## Limitations

* Many rare classes remain underrepresented (MIN_SAMPLES_PER_CLASS = 3)
* No baseline correction or smoothing applied to raw spectra
* 1D CNN requires significant compute and TensorFlow dependency
* Spectral noise not explicitly modeled or handled
* Hyperparameter tuning performed minimally (production use should optimize per-model)

---

## Possible Improvements

## Data

* Increase `MIN_SAMPLES_PER_CLASS` to reduce class imbalance
* Remove noisy spectra via SNR filtering
* Apply baseline correction/smoothing (Savitzky-Golay, etc.)
* Class weighting or synthetic data generation (SMOTE)

## Features

* Peak extraction (local maxima, FWHM)
* Spectral smoothing preprocessing
* Derivative spectra (1st/2nd order)
* Wavelet transform features

## Models

* XGBoost (gradient boosting alternative)
* Transformer encoder for spectral sequences
* Ensemble voting across multiple models
* Hyperparameter grid search or Bayesian optimization

## Evaluation

* Confusion matrix visualization
* Top-k accuracy (top 5/10 predictions)
* Per-class precision/recall/F1
* Class grouping by mineral family
* Cross-validation for robustness

---

## Final Conclusion

This pipeline successfully transforms raw Raman spectroscopy data into a machine-learning-ready dataset and compares 6 distinct model architectures.

### Key Achievements

* Robust data parsing and cleaning (1000s of files)
* Smart feature alignment via interpolation (variable-length → 801-feature vectors)
* Effective per-spectrum normalization
* **6 working multi-class classifiers** with diverse approaches:
  - 2 tree-based models (Random Forest, Extra Trees)
  - 2 linear models (Logistic Regression, Linear SVM)
  - 1 distance-based model (KNN)
  - 1 deep learning model (1D CNN)
* Automated benchmarking and model ranking (Step 7)

### Dataset Properties

* Input: 9,815 raw spectra across 2,302 mineral classes
* After filtering: 7,815 spectra across 1,274 classes
* Feature space: 801-dimensional (400–1200 cm⁻¹ wavenumber range)
* Train/test: stratified split with 79%/21% ratio

### Next Steps for Production Use

1. Run `step07_compare_models.py` to identify best model
2. Select top-performing model script from Step 6
3. Implement hyperparameter tuning for chosen architecture
4. Apply data improvements (smoothing, baseline correction, class balancing)
5. Validate on held-out test set or cross-validation
6. Deploy or integrate into mineral classification system

**The dataset is confirmed to be trainable, and the modular design enables rapid experimentation with new model architectures.**

---
