# Project State: Holomine Property Price Prediction From Sales Desc Task 2

**Schema version:** 1.1
**Last updated:** 2026-09-20
**Status:** In Progress
**Current phase:** Planning
**Project type:** PREDICTIVE
**Decision owner:** User
**Primary notebook:** hology exp1.ipynb
**Last completed cell:** Cell 7
**Data snapshot:** Kaggle competition release (2026-09-20)
**Code version:** 652d6d8

## Objective and Success Criteria

**Problem or decision:** Predict property listing prices (`listPrice`) from unstructured property sales descriptions (`text`).

**Unit of analysis:** One property listing.

**Outcome or target:** `listPrice` (numeric listing price).

**Primary success metric:** Mean Absolute Error (MAE), lower is better.

**Constraints:** Competition duration (~10 hours remaining); text is the only explanatory feature; offline submission evaluated against hidden test labels.

## Data Inventory

| Source | Metadata / Grain | Time Coverage | Sensitivity | Exploration Status |
|---|---|---|---|---|
| train.csv | 14,640 rows, 3 columns (`id`, `text`, `listPrice`), 0 missing | Unknown | Public competition data | Cell 1 through Cell 7 |
| test.csv | 3,659 rows, 2 columns (`id`, `text`), 0 missing | Unknown | Public competition data | Cell 1, Cell 3, Cell 4, Cell 7 |
| sample_submission.csv | 3,659 rows, 2 columns (`id`, `listPrice`) | Unknown | Public competition data | Cell 1 |

## Progress

| Stage | Status | Evidence / Cell | Notes |
|---|---|---|---|
| Planning | In Progress | User prompt | Problem defined, evaluation metric confirmed as MAE |
| Data quality and preparation | Completed | Cell 1 | Verified shapes, 80/20 train-test ratio, 0 nulls, unique IDs |
| EDA or statistical analysis | Completed | Cell 2 through Cell 7 | Distribution, regex spec yields, lexical drivers, archetypes, and geo profiles complete |
| Feature engineering | In Progress | Cell 4, Cell 6, Cell 7 | Archetypes, physical specs, geographic markers, and TF-IDF ready |
| Modeling and validation | Not Started | Filesystem metadata | Cross-validation scheme and baseline models to be established |
| Explanation and error analysis | Not Started | Filesystem metadata | Pending initial model outputs |
| Methodology design | Not Applicable | Problem framing | Standard text regression workflow |
| Delivery and export QA | Not Started | Filesystem metadata | Submission format confirmed from sample_submission.csv |
| Operational monitoring | Not Applicable | Problem framing | Offline competition context |

## Assumptions Register

| ID | Assumption | Category | Confidence | Impact if Wrong | Validation Plan | Status |
|---|---|---|---|---|---|---|
| A-001 | Text descriptions contain extractable physical property attributes (e.g. area, rooms, location) alongside marketing prose | Data | High | High | Sample text inspection and regex pattern validation in initial EDA | Validated |
| A-002 | Target listPrice distribution contains skewness and wide price ranges typical of real estate listings | Statistical | High | High | Compute summary statistics and distribution percentiles on train set | Validated |
| A-003 | Test set text distribution matches training set without severe covariate shift | Data | High | High | Compare vocabulary overlap, text lengths, and distribution checks | Validated |
| A-004 | Evaluation strictly computes Mean Absolute Error across all 3,659 test rows | Business | High | High | Verified against competition rules and sample_submission.csv | Validated |

## Decisions Log

### D-001 — Target Evaluation Metric Alignment

- **Chosen:** Mean Absolute Error (MAE) as primary optimization and cross-validation metric.
- **Alternatives considered:** RMSE, MAPE, RMSLE.
- **Why:** Competition evaluation explicitly measures performance using MAE.
- **Evidence:** User-provided competition specification.
- **Revisit when:** Not applicable.

### D-002 — Dual Text Representation Strategy

- **Chosen:** Combine structured feature extraction (extracting physical specs like area, rooms, location via regex/heuristics) with statistical text features (TF-IDF, n-grams) and dense text representations.
- **Alternatives considered:** Pure end-to-end transformer without tabular extraction, or pure regex extraction ignoring free text.
- **Why:** Real estate descriptions contain concrete numerical attributes embedded within variable free-text descriptions.
- **Evidence:** Competition task description stating participants must transform unstructured text into property value estimates.
- **Revisit when:** Initial exploratory analysis reveals whether key attributes are consistently present in the text.

### D-003 — Property Archetype Segmentation

- **Chosen:** Engineer explicit indicator features separating vacant land/lots from constructed residential homes and condominiums.
- **Alternatives considered:** Relying solely on bag-of-words without explicit archetype flags.
- **Why:** Vocabulary correlation reveals vacant land tokens ('lot', 'parcel', 'land') and luxury residential tokens ('suite', 'marble', 'chef kitchen') represent distinct price regimes.
- **Evidence:** Cell 5 correlation findings where land tokens show strong negative correlation (-0.24) and luxury residential shows strong positive correlation (+0.29).
- **Revisit when:** Validation of archetype segmentation in Cell 6.

### D-004 — Geographic and Market Feature Extraction

- **Chosen:** Parse state identifiers (Oregon concentration vs. high-cost New York / California metros) and regional keywords as tabular features.
- **Alternatives considered:** Relying purely on unstructured bag-of-words for geography.
- **Why:** Regional analysis reveals ~64% Oregon concentration and ~4% New York metro with a ~3x price premium (mean $2.23M vs $814k).
- **Evidence:** Cell 7 geographic distribution and price breakdown.
- **Revisit when:** Evaluating feature importance in tree-based model baseline.

## Evidence Ledger

| ID | Claim or Result | Value | Evidence Source | Evaluation Context | Status |
|---|---|---|---|---|---|
| E-001 | Test set row count | 3,659 rows | sample_submission.csv metadata | Holdout | Observed |
| E-002 | Raw dataset file inventory on disk | train.csv (14.50 MB), test.csv (3.80 MB), sample_submission.csv (47.58 KB) | Filesystem directory listing | Descriptive | Observed |
| E-003 | Train/test structural profiles and missingness | train: (14640, 3), test: (3659, 2), 0 nulls, 100% unique IDs | Cell 1 | Descriptive | Observed |
| E-004 | Target listPrice summary and naive baseline | min: 1.00, max: 80M, median: 499,900, mean: 839,073, skewness: 15.29, naive median-predictor MAE: 550,252.20 | Cell 2 | Train | Observed |
| E-005 | Text length and content distribution | train median 135 words (max 647), test median 142 words (max 624); English US listings with [Redacted Entity] | Cell 3 | Descriptive | Observed |
| E-006 | Regex specification extraction yield and correlation | beds 25.8% (corr +0.275), baths 19.1% (corr +0.459), sqft 16.6% (corr +0.139), acres 14.2% (+0.170), garage 4.4% (+0.295); test yields match within 1-2% | Cell 4 | Descriptive | Observed |
| E-007 | Lexical price drivers | Higher price: suite (+0.289), primary suite (+0.224), marble (+0.236), chef (+0.226). Lower price: lot (-0.245), build (-0.220), parcel (-0.182), utilities (-0.158) | Cell 5 | Descriptive | Observed |
| E-008 | Property archetype price regimes | Vacant Land (20.6%, median 350k, mean 558k), Condo (8.2%, median 425k, mean 1.09M), Single-Family (64.7%, median 525k, mean 718k), Luxury (6.4%, median 1.10M, mean 2.64M) | Cell 6 | Train | Observed |
| E-009 | Regional price divergence and market distribution | OR market (63.6% train / 60.9% test, median 509k), NY market (3.2% train / 4.5% test, median 761k, mean 2.23M), Unidentified (31.9% train / 32.8% test, median 475k) | Cell 7 | Descriptive | Observed |

## Methodology Design

**Method ID and version:** Not Applicable

**Maturity:** Not Applicable

**Method spec:** Not Applicable

**Case signature and baseline gap:** Not Applicable

**Exact next experiment:** Not Applicable

## Model Drivers and Error Analysis

Not applicable at planning stage. No models trained yet.

## Artifacts

| Artifact | Purpose | Validation Status |
|---|---|---|
| PROJECT_STATE.md | Central project state, evidence tracking, and audit trail | Passed |

## Operational Monitoring

**Status:** Not Applicable

**Owner and cadence:** Not Applicable

**Signals and intervention thresholds:** Not Applicable

**Fallback or rollback:** Not Applicable

## Limitations and Risks

- Text variability: Inconsistent abbreviations, missing attributes, or mixed language in property listings.
- Target skew: Outlier luxury properties could distort models if loss functions or scaling are misaligned.
- Execution window: Approximately 10 hours remain in competition, requiring fast, reproducible validation before pursuing heavy models.

## Open Questions

- [ ] What is the dominant language and formatting structure of the property descriptions?
- [ ] Are there extreme outliers, non-positive values, or abnormal entries in target listPrice?
- [ ] What compute resources (GPU / CPU cores) are available for feature extraction and model training?

## Exact Next Action

Begin exploratory data analysis and data quality verification (Cell 1: inspect dataset dimensions, missing values, and column data types without printing raw target values indiscriminately).
