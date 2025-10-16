## Validation of Histopathology Foundation Models (Hist_FMs)

This repository provides modularized Python scripts for **benchmarking and validating histopathology foundation models (FMs)** on downstream tasks such as feature extraction, linear probing, and k-nearest neighbor (kNN) evaluation.
It is designed to work seamlessly with pre-trained models (e.g., DINO, Virchow, UNI, PLUTO, etc.) and custom datasets across multiple histological stains (H&E, PAS, Silver, TRI).

---

### Repository Structure

```
Github code/
│
├── Extract_FE_modularized.py   # Feature extraction from WSIs or tiles using pretrained FMs
├── kNN_probing.py              # k-Nearest Neighbor probing for representation quality evaluation
├── linear_probing.py           # Linear classifier training for frozen feature evaluation
├── model_builders.py           # Model construction utilities for different architectures
├── utils.py                    # Helper functions for data handling, logging, and preprocessing
├── vision_transformer.py       # Custom Vision Transformer backbone definitions or wrappers
└── requirements.txt            # List of Python dependencies
```

---

### Overview of the Pipeline

1. **Feature Extraction** (`Extract_FE_modularized.py`):

   * Extracts embeddings from WSIs or image tiles using pre-trained histopathology foundation models.
   * Outputs features in `.pt` format for downstream analysis.

2. **Feature Evaluation**

   * **Linear Probing** (`linear_probing.py`): Trains a shallow linear classifier on frozen embeddings to assess feature separability.
   * **kNN Probing** (`kNN_probing.py`): Performs non-parametric evaluation to measure representational robustness.

3. **Model Management**

   * **Model Builders** (`model_builders.py`): Handles model loading (ViT, DINOv2, etc.) and configuration.
   * **Vision Transformer** (`vision_transformer.py`): Implements or extends transformer backbones with histopathology-specific modifications.

4. **Utilities**

   * **utils.py**: Contains reusable helper functions for file I/O, metrics computation, dataset organization, and visualization.

---

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/Validation_of_Hist_FMs_new.git
cd Validation_of_Hist_FMs_new/Github\ code

# (Recommended) Create and activate a conda environment
conda create -n hist_fm python=3.10 -y
conda activate hist_fm

# Install dependencies
pip install -r requirements.txt
```

---

### Usage

Example workflow for feature extraction and probing:

```bash
# Step 1: Extract features
python Extract_FE_modularized.py \
    --input_dir /path/to/tiles \
    --output_dir /path/to/features \
    --model_name DINOv2_Large

# Step 2: Run linear probing
python linear_probing.py \
    --features_dir /path/to/features \
    --labels_csv /path/to/labels.csv

# Step 3: Run kNN probing
python kNN_probing.py \
    --features_dir /path/to/features \
    --labels_csv /path/to/labels.csv
```

---

### Supported Models

* DINO / DINOv2 / DINOv3
* ViT-Base / ViT-Large / ViT-Huge
* Virchow, UNI, PLUTO, Hibou, SP22M, SP85M (custom FM checkpoints)
* Any compatible PyTorch-based vision backbone

---

### Outputs

* Extracted feature tensors (`.pt`)
* Linear probing accuracy / loss CSV files
* kNN probing metrics (accuracy, precision, recall, F1)
* Log files for each experiment

---

### Author & Contact

**Harishwar Reddy Kasireddy**
Ph.D. Candidate, University of Florida
Sarder Lab – Intelligent Critical Care Center
📧 [harishwarreddy.k@ufl.edu](mailto:harishwarreddy.k@ufl.edu)

---

## Feature Extraction — `Extract_FE_modularized.py`

###  Overview

`Extract_FE_modularized.py` performs **sequential feature extraction** from histopathology image tiles or patches using a registry of **foundation models (FMs)** such as **UNI**, **Virchow**, **Hibou**, **SP22M**, **Prov-Gigapath**, and others.
It is designed for **VRAM efficiency** — loading one model at a time, extracting embeddings for all images, saving them, and then unloading the model before moving to the next.

This script serves as the **first stage** of the histopathology FM validation pipeline, generating `.pt` feature tensors for downstream linear or kNN probing.

---

### Key Features

* Supports **multiple foundation models**: UNI, Virchow, Hibou, H-optimus, SP22M, SP85M, Prov-Gigapath, etc.
* **Sequential extraction** — builds one model at a time to minimize GPU memory usage.
* Fully **deterministic and reproducible**: TF32 disabled, fixed cuDNN behavior, deterministic GEMMs.
* **Automatic directory creation** and safe skipping of existing `.pt` files.
* Compatible with CUDA or CPU fallback.
* Configurable from the command line with flexible argument parsing.

---

### Input and Output Structure

#### **Input:**

A directory tree of tiles or patches:

```
/path/to/tiles/
├── slide_001/
│   ├── tile_0.png
│   ├── tile_1.png
│   └── ...
├── slide_002/
│   ├── tile_0.png
│   └── ...
```

#### **Output:**

For each selected model, a parallel directory containing per-image feature tensors:

```
/path/to/output/
├── UNI/
│   ├── slide_001/
│   │   ├── tile_0.pt
│   │   ├── tile_1.pt
│   └── ...
├── Virchow/
│   ├── slide_001/
│   ├── ...
```

Each `.pt` file contains a single PyTorch tensor corresponding to the feature embedding extracted by the model.

---

### Command-Line Arguments

| Argument          | Type                   | Description                                                                  |
| ----------------- | ---------------------- | ---------------------------------------------------------------------------- |
| `--input_root`    | `str`                  | Path to the input directory containing all image tiles.                      |
| `--output_base`   | `str`                  | Root directory where feature folders for each model will be created.         |
| `--models`        | `list[str]` (optional) | Space-separated list of models to extract (default: all models in registry). |
| `--skip_existing` | `flag`                 | Skip already processed images (default: `True`).                             |

---

### Supported Models

| Category             | Models                       |
| -------------------- | ---------------------------- |
| **UNI Family**       | `UNI`, `UNI2-h`              |
| **Virchow Family**   | `Virchow`, `Virchow2`        |
| **Hibou Family**     | `Hibou-B`, `Hibou-L`         |
| **H-Optimus Family** | `H-optimus-0`, `H-optimus-1` |
| **SP Models**        | `SP22M`, `SP85M`             |
| **Other**            | `Prov-Gigapath`              |

---

### Example Usage

#### **Extract features for all models**

```bash
python Extract_FE_modularized.py \
    --input_root /orange/pinaki.sarder/harishwarreddy.k/Datasets/LN_PAS_Tiles \
    --output_base /orange/pinaki.sarder/harishwarreddy.k/Validation_FM_Features \
    --skip_existing
```

#### **Extract features for selected models only**

```bash
python Extract_FE_modularized.py \
    --input_root /orange/pinaki.sarder/harishwarreddy.k/Datasets/LN_PAS_Tiles \
    --output_base /orange/pinaki.sarder/harishwarreddy.k/Validation_FM_Features \
    --models UNI Virchow SP22M \
    --skip_existing
```

#### **Example Output Directory Tree**

```
Validation_FM_Features/
├── UNI/
├── Virchow/
├── SP22M/
└── ...
```

---

### Notes

* Each model builder function (e.g., `build_UNI`, `build_Virchow`) is defined in [`model_builders.py`](./model_builders.py).
* The script automatically resets CUDA cache after each model (`reset_cuda()`).
* You can safely interrupt and resume extraction; processed files will be skipped.
* Output `.pt` tensors are ready for downstream use with [`linear_probing.py`](./linear_probing.py) or [`kNN_probing.py`](./kNN_probing.py).

---

Would you like me to generate **similar per-file README sections next** for
`linear_probing.py`, `kNN_probing.py`, and `model_builders.py` (so that all fit together cleanly in your repo)?

---

## Linear Probing — `linear_probing.py`

### Overview

`linear_probing.py` evaluates pre-extracted **foundation model (FM) embeddings** using **logistic regression** with stratified group cross-validation and bootstrap confidence intervals.
It provides a simple yet powerful way to assess how linearly separable the embeddings are for each foundation model (e.g., UNI, Virchow, Hibou, SP22M, Prov-Gigapath, etc.).

The pipeline includes:

* Stratified **Group K-Fold cross-validation** (to avoid group leakage across slides/patients).
* Aggregation of metrics (accuracy, F1, AUROC, MCC, etc.).
* **Bootstrap resampling** to estimate 95% confidence intervals.
* Result outputs in `.csv` format for each evaluated foundation model.

---

### Expected CSV File Structure

Your input CSV must contain the following **three columns**:

| Column Name | Description                                                                                     | Example                          |
| ----------- | ----------------------------------------------------------------------------------------------- | -------------------------------- |
| `ID`        | Filename (without or with image extension) corresponding to the embedding `.pt` file.           | `slide_001_tile_12.png`          |
| `Group_ID`  | Identifier for grouping samples (e.g., slide ID, patient ID). Used in grouped cross-validation. | `Slide_A`                        |
| `class`     | Target label for classification (integer or string).                                            | `0` / `1` / `disease` / `normal` |

#### Example `metadata.csv`

```csv
ID,Group_ID,class
tile_001.png,Slide_A,0
tile_002.png,Slide_A,1
tile_003.png,Slide_B,0
tile_004.png,Slide_B,1
```

#### Folder Structure

```
/path/to/embeddings/
├── UNI/
│   ├── tile_001.pt
│   ├── tile_002.pt
│   └── ...
├── Virchow/
│   ├── tile_001.pt
│   ├── tile_002.pt
│   └── ...
└── ...
```

Each `.pt` file should contain a single **1D PyTorch tensor** representing the feature embedding.

---

### Command-Line Arguments

| Argument      | Type        | Description                                                                             |
| ------------- | ----------- | --------------------------------------------------------------------------------------- |
| `--csv_file`  | `str`       | Path to the CSV file containing columns: `ID`, `Group_ID`, `class`.                     |
| `--emb_dir`   | `str`       | Root directory containing per-model embedding folders (e.g., UNI, Virchow, etc.).       |
| `--out_dir`   | `str`       | Directory where result CSVs will be saved.                                              |
| `--models`    | `list[str]` | (Optional) Space-separated list of model names to evaluate. Default: all supported FMs. |
| `--splits`    | `int`       | Number of StratifiedGroupKFold splits (default: `5`).                                   |
| `--seeds`     | `list[int]` | Random seeds for reproducibility (default: `[0, 1, 2]`).                                |
| `--bootstrap` | `int`       | Number of bootstrap replicates for confidence interval estimation (default: `1000`).    |

---

### Output Files (per model)

Each evaluated model produces **three output files** under the specified `--out_dir`.

| File                                  | Description                              |
| ------------------------------------- | ---------------------------------------- |
| `cv_results_LR_<model>.csv`           | Per-fold cross-validation metrics.       |
| `bootstrap_replicates_LR_<model>.csv` | Raw bootstrap metric replicates.         |
| `bootstrap_CI_LR_<model>.csv`         | Mean and 95% CI summary for each metric. |

#### Example Output Structure

```
results/
├── UNI/
│   ├── cv_results_LR_UNI.csv
│   ├── bootstrap_replicates_LR_UNI.csv
│   └── bootstrap_CI_LR_UNI.csv
├── Virchow/
│   ├── cv_results_LR_Virchow.csv
│   ├── bootstrap_replicates_LR_Virchow.csv
│   └── bootstrap_CI_LR_Virchow.csv
└── ...
```

---

### Example Commands

#### Run evaluation for **all foundation models**

```bash
python linear_probing.py \
    --csv_file /orange/pinaki.sarder/harishwarreddy.k/Datasets/metadata.csv \
    --emb_dir /orange/pinaki.sarder/harishwarreddy.k/Validation_FM_Features \
    --out_dir /orange/pinaki.sarder/harishwarreddy.k/LinearProbing_Results \
    --splits 5 \
    --bootstrap 1000
```

#### Run evaluation for **selected models only**

```bash
python linear_probing.py \
    --csv_file /orange/pinaki.sarder/harishwarreddy.k/Datasets/metadata.csv \
    --emb_dir /orange/pinaki.sarder/harishwarreddy.k/Validation_FM_Features \
    --out_dir /orange/pinaki.sarder/harishwarreddy.k/LinearProbing_Results \
    --models UNI Virchow SP22M \
    --splits 5 \
    --bootstrap 500
```

---

### Metrics Computed

#### **Binary Classification**

* Accuracy
* Balanced Accuracy
* Matthews Correlation Coefficient (MCC)
* F1, Precision, Recall, Specificity
* AUROC, AUPRC

#### **Multiclass (Macro-Averaged)**

* F1_macro, Precision_macro, Recall_macro
* Specificity_macro
* AUROC_macro, AUPRC_macro

Each metric includes **mean estimates** and **95% confidence intervals** from bootstrapping.

---

### Notes

* Designed to evaluate embeddings extracted using [`Extract_FE_modularized.py`](#-feature-extraction--extract_fe_modularizedpy).
* Fully **deterministic and reproducible**: random seeds are fixed and `torch.use_deterministic_algorithms(True)` is enabled.
* Handles missing `.pt` files gracefully with warnings.
* Uses **StratifiedGroupKFold** to prevent leakage between slides or patients.
* Logistic regression regularization parameter ( C ) is defined as:
  [
  C = \frac{1}{(100 / (M \times C_{\text{classes}}))}
  ]
  where *M* is the number of features and *C_classes* is the number of unique class labels.

---

Perfect — this script is your **k-Nearest Neighbor (kNN) probing module**, used to evaluate the *representational quality* of the embeddings generated by your foundation models (FMs).

Below is the **ready-to-paste GitHub-compatible Markdown** section for your `README.md`, complete with headings (`##`, `###`, etc.), tables, and code blocks.
You can drop this **as-is** into your raw README file — GitHub will render it cleanly.

---

## kNN Probing — `kNN_probing.py`

### Overview

`kNN_probing.py` evaluates **foundation model (FM) embeddings** using a **non-parametric k-Nearest Neighbors classifier**.
It measures how well the extracted embeddings cluster by class — i.e., how “semantically separable” the representation space is — **without training a classifier**.

This is a complementary analysis to linear probing, providing insight into **representation quality** and **class neighborhood structure**.

The script performs:

* **Stratified Group K-Fold cross-validation** (to prevent leakage between related samples).
* Evaluation using kNN (default `k=20`) on standardized embeddings.
* Computation of comprehensive metrics (Accuracy, MCC, AUROC, etc.).
* **Bootstrap resampling** for 95% confidence intervals.
* Automatic per-model reporting in CSV format.

---

### Expected CSV File Structure

Your input CSV must include **three required columns**:

| Column     | Description                                                                 | Example                           |
| ---------- | --------------------------------------------------------------------------- | --------------------------------- |
| `ID`       | Image filename corresponding to the embedding `.pt` file.                   | `tile_001.png`                    |
| `Group_ID` | Group identifier (e.g., slide ID, patient ID) used for stratified grouping. | `Slide_A`                         |
| `class`    | Class label (integer or string).                                            | `0` / `1` / `abnormal` / `normal` |

#### Example `metadata.csv`

```csv
ID,Group_ID,class
tile_001.png,Slide_A,0
tile_002.png,Slide_A,1
tile_003.png,Slide_B,0
tile_004.png,Slide_B,1
```

#### Folder Structure

```
/path/to/embeddings/
├── UNI/
│   ├── tile_001.pt
│   ├── tile_002.pt
│   └── ...
├── Virchow/
│   ├── tile_001.pt
│   ├── tile_002.pt
│   └── ...
└── ...
```

Each `.pt` file contains a **1D PyTorch tensor** (embedding vector) for the corresponding image.

---

### Command-Line Arguments

| Argument      | Type        | Description                                                                     |
| ------------- | ----------- | ------------------------------------------------------------------------------- |
| `--csv_file`  | `str`       | Path to CSV with columns `ID`, `Group_ID`, `class`.                             |
| `--emb_dir`   | `str`       | Directory containing per-model embedding subfolders (e.g., UNI, Virchow, etc.). |
| `--out_dir`   | `str`       | Directory where output CSVs will be stored.                                     |
| `--models`    | `list[str]` | Optional list of FMs to run. Default: all available models.                     |
| `--splits`    | `int`       | Number of StratifiedGroupKFold splits (default: `5`).                           |
| `--seeds`     | `list[int]` | Random seeds for reproducibility (default: `[0, 1, 2]`).                        |
| `--bootstrap` | `int`       | Number of bootstrap iterations (default: `1000`).                               |

---

### Output Files (per model)

Each evaluated foundation model produces **three CSV files** inside the specified `--out_dir`.

| File                                   | Description                                                           |
| -------------------------------------- | --------------------------------------------------------------------- |
| `cv_results_kNN_<model>.csv`           | Metrics from each cross-validation fold.                              |
| `bootstrap_replicates_kNN_<model>.csv` | Raw metrics from all bootstrap runs.                                  |
| `bootstrap_CI_kNN_<model>.csv`         | Summary table with mean and 95% confidence intervals for each metric. |

#### Example Output Directory

```
results/
├── UNI/
│   ├── cv_results_kNN_UNI.csv
│   ├── bootstrap_replicates_kNN_UNI.csv
│   └── bootstrap_CI_kNN_UNI.csv
├── Virchow/
│   ├── cv_results_kNN_Virchow.csv
│   ├── bootstrap_replicates_kNN_Virchow.csv
│   └── bootstrap_CI_kNN_Virchow.csv
└── ...
```

---

### Example Commands

#### Run kNN probing for **all foundation models**

```bash
python kNN_probing.py \
    --csv_file /orange/pinaki.sarder/harishwarreddy.k/Datasets/metadata.csv \
    --emb_dir /orange/pinaki.sarder/harishwarreddy.k/Validation_FM_Features \
    --out_dir /orange/pinaki.sarder/harishwarreddy.k/kNN_Results \
    --splits 5 \
    --bootstrap 1000
```

#### Run kNN probing for **specific models only**

```bash
python kNN_probing.py \
    --csv_file /orange/pinaki.sarder/harishwarreddy.k/Datasets/metadata.csv \
    --emb_dir /orange/pinaki.sarder/harishwarreddy.k/Validation_FM_Features \
    --out_dir /orange/pinaki.sarder/harishwarreddy.k/kNN_Results \
    --models UNI Virchow SP22M \
    --splits 5 \
    --bootstrap 500
```

---

### Metrics Computed

#### **Binary Classification**

* Accuracy
* Balanced Accuracy
* Matthews Correlation Coefficient (MCC)
* F1, Precision, Recall, Specificity
* AUROC, AUPRC

#### **Multiclass (Macro-Averaged)**

* F1_macro, Precision_macro, Recall_macro
* Specificity_macro
* AUROC_macro, AUPRC_macro

Each metric includes **mean estimates** and **bootstrap-derived 95% confidence intervals**.

---

### Algorithm Details

* Uses **k-Nearest Neighbors (kNN)** classifier (`n_neighbors=20`) with standardized embeddings.
* Each fold performs **feature standardization (Z-score normalization)** via `StandardScaler`.
* If any class is missing in a fold, probability vectors are **zero-padded** to preserve output shape.
* Ensures **deterministic execution** with fixed seeds and `torch.use_deterministic_algorithms(True)`.
* Bootstrap CI computed using **group-level resampling** to account for correlated samples.

---

### Notes

* Designed to evaluate embeddings generated via [`Extract_FE_modularized.py`](#-feature-extraction--extract_fe_modularizedpy).
* The kNN probing analysis helps quantify **clustering strength** and **representation smoothness** of embeddings.
* Larger gaps between linear and kNN accuracy indicate **non-linear class separation** in embedding space.
* You can modify `n_neighbors` in `evaluate_knn()` (default: `20`) for sensitivity analysis.

---
Perfect — this new script (`linear_regression.py`) is a **ridge regression–based evaluation pipeline** for multi-output regression tasks (e.g., predicting cell type proportions or continuous histopathology features) using foundation model embeddings.

Below is a **ready-to-paste GitHub Markdown section** for your `README.md`, consistent with the earlier style.
You can copy it **as-is** into your raw README file — it will render properly on GitHub.

---

## Ridge Regression — `linear_regression.py`

### Overview

`linear_regression.py` performs **ridge regression–based evaluation** on foundation model (FM) embeddings to assess their predictive power for **continuous or multi-output targets** (e.g., cell type proportions, molecular scores, or morphometric features).

It uses **grouped cross-validation** and **bootstrap confidence intervals** to provide robust regression performance estimates.
This complements linear and kNN probing, extending the evaluation to **quantitative prediction tasks**.

The pipeline includes:

* Group-aware K-fold regression using `Ridge` from scikit-learn
* Metrics such as R², MAE, RMSE, MAPE, and mean Pearson correlation
* Bootstrap resampling to estimate 95% confidence intervals
* Automatic aggregation of results per foundation model

---

### Expected CSV File Structure

Your CSV file must contain **two required columns**:

| Column     | Description                                                                                                  | Example            |
| ---------- | ------------------------------------------------------------------------------------------------------------ | ------------------ |
| `ID`       | Embedding filename (without or with extension). Must match `.pt` files in the embeddings and labels folders. | `Slide01_0001.png` |
| `Group_ID` | Group identifier (e.g., slide ID, patient ID). Used for grouped cross-validation.                            | `Slide01`          |

#### Example `metadata.csv`

```csv
ID,Group_ID
Slide01_0001.png,Slide01
Slide01_0002.png,Slide01
Slide02_0001.png,Slide02
Slide02_0002.png,Slide02
```

---

### Folder Structure

#### Input directories:

```
/path/to/embeddings/
├── UNI/
│   ├── Slide01_0001.pt
│   ├── Slide01_0002.pt
│   └── ...
├── Virchow/
│   ├── Slide02_0001.pt
│   ├── Slide02_0002.pt
│   └── ...
└── ...

/path/to/labels/
├── Slide01_0001.pt
├── Slide01_0002.pt
├── Slide02_0001.pt
└── Slide02_0002.pt
```

Each embedding (`.pt`) file should contain a **1D feature vector** (from a foundation model).
Each corresponding label (`.pt`) file should contain a **vector of continuous values** — e.g., predicted cell type proportions or regression targets.

---

### Command-Line Arguments

| Argument            | Type        | Description                                                                       |
| ------------------- | ----------- | --------------------------------------------------------------------------------- |
| `--embeddings_dir`  | `str`       | Root directory containing per-model embedding folders (e.g., UNI, Virchow, etc.). |
| `--labels_dir`      | `str`       | Directory containing label `.pt` files with the same filenames as embeddings.     |
| `--csv_path`        | `str`       | CSV file containing columns `ID` and `Group_ID`.                                  |
| `--models`          | `list[str]` | (Optional) Foundation models to evaluate. Default: all available FMs.             |
| `--n_splits`        | `int`       | Number of cross-validation folds (default: `5`).                                  |
| `--seeds`           | `list[int]` | Random seeds for cross-validation (default: `[0, 1, 2]`).                         |
| `--bootstrap_iters` | `int`       | Number of bootstrap replicates for confidence intervals (default: `1000`).        |
| `--output_dir`      | `str`       | Directory to save regression results (default: `results_regression`).             |

---

### Output Files (per model)

Each evaluated foundation model produces **three CSV outputs** under the specified output directory.

| File                                          | Description                                         |
| --------------------------------------------- | --------------------------------------------------- |
| `regression_cv_results_<model>.csv`           | Cross-validation metrics (one row per fold).        |
| `regression_bootstrap_replicates_<model>.csv` | Raw bootstrap metrics for all iterations.           |
| `regression_bootstrap_CI_<model>.csv`         | Mean and 95% CI summary for each regression metric. |

#### Example Output Directory

```
results_regression/
├── UNI/
│   ├── regression_cv_results_UNI.csv
│   ├── regression_bootstrap_replicates_UNI.csv
│   └── regression_bootstrap_CI_UNI.csv
├── Virchow/
│   ├── regression_cv_results_Virchow.csv
│   ├── regression_bootstrap_replicates_Virchow.csv
│   └── regression_bootstrap_CI_Virchow.csv
└── ...
```

---

### Example Commands

#### Run regression for **all foundation models**

```bash
python linear_regression.py \
    --embeddings_dir /orange/pinaki.sarder/harishwarreddy.k/Validation_FM_Features \
    --labels_dir /orange/pinaki.sarder/harishwarreddy.k/CellType_Proportions \
    --csv_path /orange/pinaki.sarder/harishwarreddy.k/Datasets/metadata.csv \
    --output_dir /orange/pinaki.sarder/harishwarreddy.k/Regression_Results \
    --n_splits 5 \
    --bootstrap_iters 1000
```

#### Run regression for **specific models only**

```bash
python linear_regression.py \
    --embeddings_dir /orange/pinaki.sarder/harishwarreddy.k/Validation_FM_Features \
    --labels_dir /orange/pinaki.sarder/harishwarreddy.k/CellType_Proportions \
    --csv_path /orange/pinaki.sarder/harishwarreddy.k/Datasets/metadata.csv \
    --models UNI Virchow SP22M \
    --output_dir /orange/pinaki.sarder/harishwarreddy.k/Regression_Results \
    --n_splits 5 \
    --bootstrap_iters 500
```

---

### Metrics Computed

| Metric             | Description                                                             |
| ------------------ | ----------------------------------------------------------------------- |
| **R²**             | Coefficient of determination (variance explained).                      |
| **MAE**            | Mean Absolute Error — average magnitude of prediction error.            |
| **RMSE**           | Root Mean Squared Error — penalizes large deviations.                   |
| **MAPE**           | Mean Absolute Percentage Error (robust to scale differences).           |
| **Mean Pearson r** | Mean column-wise Pearson correlation between predicted and true values. |

Each metric includes **mean estimates** across folds and **bootstrap 95% confidence intervals**.

---

### Algorithm Details

* Uses **Ridge Regression** (`α=1.0`) for stable coefficient estimation.
* Performs **Z-score normalization** on embeddings before training.
* Splits data with **GroupShuffleSplit** to prevent leakage between related groups (e.g., same slide/patient).
* Computes **multi-output regression metrics** across all targets.
* Applies **bootstrap resampling** at the sample level to compute confidence intervals.

---

### Notes

* Designed for embeddings produced by [`Extract_FE_modularized.py`](#-feature-extraction--extract_fe_modularizedpy).
* Can be used for regression tasks like **predicting cell type proportions**, **molecular markers**, or **histological scores**.
* Embedding and label files must share the same `.pt` filename base.
* Outputs can be used to compare model predictive strength across FMs and stains.

---

Perfect — this new script (`copy_detection.py` or equivalent) evaluates **representation robustness under augmentations** by measuring **embedding similarity matching accuracy (Top-K accuracy)** between original and augmented image embeddings for multiple foundation models.

Below is a **ready-to-paste GitHub Markdown section** you can drop directly into your `README.md`.
It follows the same heading hierarchy and format as your previous sections (`linear_probing`, `kNN_probing`, `linear_regression`).

---

## Copy Detection — `copy_detection.py`

### Overview

`copy_detection.py` evaluates the **robustness and consistency** of foundation model (FM) embeddings when subjected to various image augmentations (e.g., geometric, color, noise, deformation).
It does this by computing **Top-K retrieval accuracy** — i.e., whether each original image’s embedding correctly identifies its augmented counterpart among all others.

This task quantifies **representation stability** — a crucial property for models expected to produce invariant embeddings across domain shifts, stain variations, or transformations.

---

### Folder Structure

Your base directory should have **one subfolder per model**, each containing:

```
/path/to/Copy_detection/
├── UNI/
│   ├── images/
│   │   ├── tile_001.pt
│   │   ├── tile_002.pt
│   │   └── ...
│   └── aug_images/
│       ├── geo/
│       │   ├── tile_001.pt
│       │   ├── tile_002.pt
│       │   └── ...
│       ├── noise/
│       ├── color/
│       └── deform/
├── Virchow/
│   ├── images/
│   └── aug_images/
│       ├── geo/
│       └── color/
└── ...
```

Each `.pt` file must contain a **1D PyTorch tensor** — the embedding vector corresponding to an original or augmented image.

---

### Command-Line Arguments

| Argument     | Type  | Description                                                                        |
| ------------ | ----- | ---------------------------------------------------------------------------------- |
| `--base_dir` | `str` | Path to the root folder (e.g., `Copy_detection/`) containing per-model subfolders. |
| `--out_csv`  | `str` | Path to save the output CSV summarizing top-K results.                             |

---

### What It Computes

For every foundation model and augmentation type, the script computes **Top-K retrieval accuracy** based on cosine similarity between normalized embeddings:

| Metric             | Meaning                                                                     |
| ------------------ | --------------------------------------------------------------------------- |
| **Top-1 Accuracy** | Fraction of images where the exact augmented pair ranks 1st in similarity.  |
| **Top-3 Accuracy** | Fraction where the augmented pair appears among top-3 most similar samples. |
| **Top-5 Accuracy** | Fraction where the augmented pair appears among top-5 most similar samples. |

Higher values indicate that the model produces **stable, augmentation-invariant embeddings**.

---

### Example Command

#### Run Copy Detection for all models

```bash
python copy_detection.py \
    --base_dir /orange/pinaki.sarder/harishwarreddy.k/Copy_detection \
    --out_csv /orange/pinaki.sarder/harishwarreddy.k/Copy_detection/results/topk_accuracy.csv
```

This will automatically:

* Iterate over each model subfolder (e.g., `UNI`, `Virchow`, `SP22M`, etc.)
* Compute Top-1/3/5 accuracies for all augmentation types under `aug_images/`
* Save a summary CSV report with results.

---

### Example Output CSV

| model   | augmentation | top1_accuracy | top3_accuracy | top5_accuracy |
| :------ | :----------- | :------------ | :------------ | :------------ |
| UNI     | geo          | 0.98          | 0.99          | 1.00          |
| UNI     | color        | 0.92          | 0.96          | 0.98          |
| Virchow | noise        | 0.87          | 0.93          | 0.95          |
| SP22M   | deform       | 0.90          | 0.95          | 0.97          |

---

### How It Works

1. **Load Embeddings**
   The script loads all `.pt` tensors from `images/` and each subfolder of `aug_images/`.

2. **Normalize Embeddings**
   Each embedding vector is L2-normalized:
   [
   v_i' = \frac{v_i}{|v_i|}
   ]

3. **Compute Similarities**
   A cosine similarity matrix between original and augmented embeddings is computed:
   [
   S = V_{orig} \cdot V_{aug}^T
   ]

4. **Rank Matching Pairs**
   For each image *i*, it checks where the corresponding augmented embedding ranks among all similarities.

5. **Aggregate Metrics**
   Top-1, Top-3, and Top-5 accuracies are averaged across all images for each augmentation type.

---

### Notes

* Embedding filenames in `images/` and each `aug_images/<type>/` folder **must match exactly** (e.g., `tile_001.pt` ↔ `tile_001.pt`).
* The script automatically validates filename consistency and reports missing pairs.
* Designed for evaluating FM embeddings produced via [`Extract_FE_modularized.py`](#🧩-feature-extraction--extract_fe_modularizedpy).
* Useful for comparing invariance of different models (e.g., UNI vs Virchow vs Hibou).
* You can modify `topk_list` in the function `compute_topk_accuracy()` to compute custom Top-K values (e.g., `[1,5,10]`).

---

### Interpretation Tips

| Observation                           | Interpretation                                                                                |
| ------------------------------------- | --------------------------------------------------------------------------------------------- |
| **High Top-1 accuracy**               | Model is strongly invariant to augmentations; embeddings are almost identical post-transform. |
| **Large gap between Top-1 and Top-5** | Minor distortions or moderate sensitivity to augmentation noise.                              |
| **Low Top-K across augmentations**    | Embeddings change significantly — model less robust to stain/augmentation shifts.             |

---