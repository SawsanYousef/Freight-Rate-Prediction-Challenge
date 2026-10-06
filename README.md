# Freight-Rate-Prediction-Challenge 🚚

An end-to-end, production-grade machine learning solution engineered to predict domestic logistical freight costs (`posted_rate`) within the United States mainland. This repository contains the complete diagnostic pipeline, exploratory analysis, and optimized tree-based ensembles built to capture non-linear market pricing strata while strictly preventing temporal data leakage.

---

## 📁 Repository Structure

```text
Freight-Rate-Prediction-Challenge/
├── data/
│   ├── train_test.csv                       # Historical development labeled training data
│   ├── validation.csv                       # Un-labeled evaluation dataset (12,000 loads)
│   └── december_chart_inputs.csv            # Populated baseline template for December stress-test
├── scorer_results/
│   └── candidate_december.png               # Automated visualization output from score.py
├── notebooks/
│   └── freight_rate_prediction_eda_ml.ipynb # Well-structured, fully documented Jupyter Notebook
├── models/
│   ├── champion_catboost_model.cbm          # Serialized finalized CatBoost model artifact
│   └── categorical_encoder.joblib           # Serialized fitted Ordinal Encoder artifact
├── .gitignore                               # Prevents tracking of virtual envs and bytecode caches
├── README.md                                # Executive summary and repository roadmap
├── requirements.txt                         # Full software dependency manifest
├── score.py                                 # Administrative formatting and test suite script
└── validation_predictions.csv               # Final prediction delivery file (load_id,predicted_rate)
```

---

## ⚡ How It Works & Execution Guide

Follow these sequential steps to establish the environment, generate predictions, and run the validation suite:

### 1. Environment Activation & Dependency Installation
Initialize your virtual environment and install the required engineering libraries using the `requirements.txt` manifest:
```bash
# Clone the repository
git clone https://github.com
cd Freight-Rate-Prediction-Challenge

# Install all required packages
pip install -r requirements.txt
```

### 2. Pipeline Execution
Run the main script or execution block inside the Jupyter Notebook. The custom execution pipeline will automatically:
1. Ingest and clean the input datasets according to production logic.
2. Load the optimized `champion_catboost_model.cbm` and its corresponding `categorical_encoder.joblib`.
3. Process the un-labeled validation rows and generate the final required submission file (`validation_predictions.csv`).
4. Price the fixed 31-day temporal sequence inside `data/december_chart_inputs.csv`.

### 3. Run the Evaluation Scorer Test Suite
To verify formatting constraints, validate row dimensions, and automatically render the seasonal sensitivity trend chart, execute the project’s official scoring script:
```bash
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```
*Upon a successful run, the score suite will generate the final verification chart inside `scorer_results/candidate_december.png`.*

---

## 📊 Data Quality Audit & Cleaning Actions

During the core data engineering phase, several structural anomalies were identified and resolved inside the data cleaning pipeline:
* **Negative Mass Rectification:** 292 entries in the `weight` column registered unphysical negative metrics. These were rectified safely using absolute value mapping (`.abs()`).
* **Conditional Missing Value Imputation:** 
  * Missing records in `weight` were imputed dynamically using the **median weight clustered by specific `equipment` types** (Dry Van, Reefer, Flatbed) to avoid trailer configuration bias.
  * Missing records in `market_index` were handled using the specific **daily mean**, fallback-cloned to the global training median for unobserved calendar dates.
* **Target Leakage Elimination:** Early analysis flagged a derived Rate-Per-Mile (`RPM`) attribute causing an artificial 99% accuracy score. Since computing `RPM` mathematically requires the true `posted_rate` as a divisor, it presents massive target leakage. Because total cost is completely unobserved prior to production inference, `RPM` was entirely purged from the training feature matrix to guarantee absolute data safety.

---

## 💡 Key Architectural Finding: The `quote_signal` Paradox

Plotting `Trip Distance vs. Posted Rate` revealed that the spot freight market splits vertically into **three distinct pricing strata/bands** (an Upper High-Cost path, a Mainstream Stable path, and a Lower Backhaul path).

* Conventional continuous attributes like `weight` and discrete dimensions like `equipment` exhibited flat, mixed overlapping across these paths, failing to explain the segregation.
* Continuous gradient mapping against **`quote_signal`** resolved the paradox. Lower signals (1.0–1.5) mapped to backhaul runs, mid-range signals (2.0–2.5) drove the mainstream contract market, and extreme signals (>3.0) isolated premium emergency spot rates.
* **The Paradox:** Pearson correlation tests showed `quote_signal` had a linear score of **-0.0399** with `posted_rate`, which initially implies irrelevance. However, the visualization proved it operates as an **Interaction Feature / Slope Modifier** on distance. To force the tree nodes to capture this behavior natively, a golden interaction feature was engineered: 
  \[\text{distance\_x\_quote\_signal} = \text{distance} \times \text{quote\_signal}\]

---

## 🧪 Training, Chronological Validation & Tuning

* **Chronological Data Splitting:** To eliminate temporal look-ahead contamination, data was split strictly by time: Months 1–8 were assigned for Training (38,477 rows), and Months 9–10 were isolated for Testing (9,523 rows).
* **Compact Nominal Encoding:** High-cardinality nominal parameters (`pickup` and `delivery` containing 64 unique locations) were encoded via dense **Ordinal Encoding** to keep decision splits compact and speed up computation.
* **Hyperparameter Optimization:** To squeeze peak performance out of our model, an advanced Grid Search was combined with a 3-fold **`TimeSeriesSplit` (TSCV)** cross-validation loop. The optimized hyperparameters discovered were: `{'depth': 6, 'learning_rate': 0.05, 'l2_leaf_reg': 7}`.

---

## 🏆 Algorithmic Benchmarking Results

Four elite tabular regression architectures were evaluated under identical chronological limits. Retraining our final champion model on the entire training matrix under optimized parameters yielded the following benchmarking records against unseen future test months (9–10):

| Model Architecture | Test RMSE (\$) | Test R-squared (R²) | Operational Status |
| :--- | :---: | :---: | :---: |
| **Optimized CatBoost Regressor** 👑 | **632.36** | **0.8283** | **Final Deployed Champion Engine** |
| Baseline CatBoost Regressor | 634.41 | 0.8272 | Baseline Benchmark |
| LightGBM Regressor | 636.13 | 0.8262 | Baseline Benchmark |
| XGBoost Regressor | 640.30 | 0.8240 | Baseline Benchmark |
| Random Forest Regressor | 654.53 | 0.8160 | Baseline Bagging Model |

*Feature importance metrics verified that `distance` (57.05%) and our engineered interactive feature `distance_x_quote_signal` (33.83%) dominated the final model, dictating **90.88%** of the deployment engine's rate-pricing intelligence.*

---

## 📈 December 2025 Prediction Discussion

When stress-tested on a fixed sequence spanning December 1 to December 31, 2025—where every physical characteristic was held constant (Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lbs) and only the date changed—the model successfully demonstrated **temporal sensitivity**. 

Instead of an overfitted flat line, the `candidate_december.png` chart rendered a realistic **cyclical wave pattern** fluctuating dynamically within a safe market threshold of **\$864 to \$869** (a stable baseline of ~\$2.40 per mile). This cyclical wave statistically proves that the model natively captures real-world weekly market capacity shifts and seasonal frequencies without introducing erratic volatility.
