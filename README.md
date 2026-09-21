<div align="center">
  <img src="logo.png" alt="Holomine Logo" width="120" height="120">

  <h1>Property Price Prediction from Sales Descriptions</h1>
  <p><strong>Holomine Task 2: Text-to-Price Regression with Calibrated Out-of-Fold Stacking</strong></p>

  <p align="center">
    <img src="https://img.shields.io/badge/Competition-Holomine_Task_2-blue?style=flat-square" alt="Competition">
    <img src="https://img.shields.io/badge/Metric-MAE-orange?style=flat-square" alt="Metric">
    <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/Best_OOF_MAE-%24333%2C753-success?style=flat-square" alt="Best OOF MAE">
    <img src="https://img.shields.io/badge/License-MIT-green?style=flat-square" alt="License">
  </p>

  <p align="center">
    An end-to-end competitive machine learning pipeline transforming unstructured real estate sales descriptions into listing price predictions through regex specification distillation, leakage-safe TF-IDF Ridge anchors, tri-model convex simplex blending, and L1 median residual calibration.
  </p>
</div>

## Tech Stack

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)
![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white)

## Problem Formulation

The competition task requires predicting the numerical listing price (`listPrice`) of residential and commercial real estate properties strictly from unstructured English sales prose (`text`). No tabular metadata columns are provided.

Evaluation is assessed using Mean Absolute Error (MAE):

$$
\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|
$$

The target distribution presents extreme structural skewness and variance:

- Training set: 14,640 property listings
- Holdout test set: 3,659 property listings
- Price range: \$1 to \$80,000,000
- Target median: \$499,900 | Target mean: \$839,073 | Skewness: 15.29
- Naive constant median baseline MAE: \$550,252.20

## Progressive Validation Milestones

All models use identical 5-fold stratified cross-validation splits partitioned across 10 quantiles of `listPrice`.

| Model Stage | Architecture | 5-Fold OOF MAE | Gain vs Baseline |
|---|---|:---:|:---:|
| Baseline | Naive Constant Median Predictor | \$550,252 | 0.00% |
| Model 1 | LightGBM Tabular (38 features) | \$459,500 | -16.5% |
| Model 2 | TF-IDF Ridge NLP Anchor | \$365,760 | -33.5% |
| Model 3 | Consolidated LightGBM (55 features) | \$344,952 | -37.3% |
| Model 4 | Tri-Model Simplex Blend + Calibration | **\$333,754** | **-39.3%** |

## System Architecture

The pipeline operates across five leakage-free stages.

**Stage 1 - Regex Tabular Distillation:** Physical specifications (`sqft`, `beds`, `baths`, `acres`, `stories`, `garage`, `year_built`) are extracted from prose using regular expressions. Structural missingness flags and spatial ratios are derived as additional features.

**Stage 2 - Lexical NLP Representation:** Over 75% of listings omit explicit numeric dimensions. Text is converted to continuous signals via out-of-fold TF-IDF vectorization (50,000 unigrams and bigrams), a Ridge price anchor (`nlp_ridge_pred`), and 32 latent semantic dimensions via TruncatedSVD.

**Stage 3 - Leakage-Free 5-Fold Cross-Validation:** All preprocessing steps (imputation, scaling, TF-IDF fitting) are isolated strictly inside training fold boundaries. Out-of-fold predictions form continuous feature inputs without target contamination.

**Stage 4 - Tri-Model Convex Simplex Optimization:** Three architecturally diverse OOF prediction vectors (LightGBM tabular, LightGBM NLP, L1-Ridge) are combined via non-negative least-squares simplex weight optimization:

$$
\min_{\mathbf{w} \ge 0} \left\| y - \sum_k w_k \hat{y}_k \right\|_1 \quad \text{subject to} \quad \sum_k w_k = 1
$$

**Stage 5 - L1 Median Residual Calibration:** Any non-zero OOF median residual introduces systematic bias under MAE. A deterministic shift is applied to the final ensemble predictions to zero out the median residual.

## Repository Structure

```
property-price-prediction-from-sales/
├── hology_notebook_1.ipynb   # Primary deliverable: full documented pipeline (31 cells)
├── README.md                 # Project overview and reproduction guide
├── logo.png                  # Holomine competition logo
├── LICENSE                   # MIT License
├── train.csv                 # Training dataset (14,640 listings)
├── test.csv                  # Test dataset (3,659 listings)
└── sample_submission.csv     # Submission format template
```

## Quickstart

The repository requires Python 3.11.

### Installation

```bash
git clone https://github.com/ababilkhoerulimam/property-price-prediction-from-sales.git
cd property-price-prediction-from-sales

python -m venv .venv
.venv\Scripts\activate

pip install numpy pandas scipy scikit-learn lightgbm matplotlib seaborn ipykernel
python -m ipykernel install --user --name property-pred-311 --display-name "Python 3.11 (Property ML)"
```

### Reproducing the Pipeline

```bash
jupyter notebook hology_notebook_1.ipynb
```

Run all cells sequentially (Cells 1 through 15) to reproduce the full data quality audit, feature extraction, 5-fold cross-validation, tri-model blending, calibration, and submission artifact exports.

### Submission Verification

```python
import pandas as pd

sub = pd.read_csv("submission_tri_enriched_cal.csv")
assert sub.shape == (3659, 2), "Invalid submission shape"
assert sub.isnull().sum().sum() == 0, "Missing values detected"
assert (sub["listPrice"] > 0).all(), "Non-positive price predictions detected"
print("Submission verified successfully.")
```

## Citation and Provenance

- Competition: [Holomine] Property Price Prediction From Sales Descriptions - Task 2
- Dataset Source: Real Estate Oregon 2026 and Real Estate New York 2026 public data collections by Kanchana Karunarathna on Kaggle Datasets