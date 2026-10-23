# Data Projects

A collection of data analysis and machine learning projects demonstrating various techniques in data science, visualization, and statistical analysis.

## Projects Overview

### 1. **Literary Text Analysis** (`literary_text_analysis.py`)
Analyzes literary texts (e.g., book chapters) to extract insights about character mentions and text structure.

**Key Features:**
- Fetches and parses text from URLs
- Splits text into chapters
- Tracks cumulative character name counts across chapters
- Analyzes chapter lengths and punctuation patterns

**Dependencies:** requests, datascience, numpy, matplotlib

**Data Files:** None (fetches from URL)

---

### 2. **Housing Market Analysis** (`housing_market_analysis.py`)
Performs exploratory data analysis and predictive modeling on housing market data.

**Key Features:**
- Data loading and preprocessing
- Feature engineering (numerical and categorical variables)
- Identification of important categorical levels
- Statistical analysis and visualization
- Uses house_train.csv dataset

**Dependencies:** pandas, numpy, matplotlib, seaborn

**Data Files:** `data/house_train.csv`

---

### 3. **Gold Price Prediction** (`gold_price_prediction.py`)
Machine learning project predicting the gold price (GLD ETF) from the S&P 500 (SPX), oil (USO) and the EUR/USD exchange rate.

**Key Features:**
- Loads and explores gold price dataset
- Correlation analysis with heatmap visualization
- Excludes SLV (silver), which tracks gold almost exactly
- Chronological train/test split and time-series cross-validation (no look-ahead leakage)
- Outlier clipping using limits learned from the training set only
- Implements multiple ML models:
  - Polynomial Regression (Lasso)
  - Random Forest Regressor
  - XGBoost Regressor
- Grid search for hyperparameter tuning
- Model evaluation using R² and RMSE, compared against a naive "last known price" baseline

**Dependencies:** pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost

**Data Files:** `data/gold_price_data.csv`

---

### 4. **Student Alcohol Correlation Analysis** (`student_alcohol_correlation.py`)
Statistical analysis exploring the relationship between student alcohol consumption and academic performance.

**Key Features:**
- Loads and cleans student data
- Computes alcohol consumption index
- Confidence interval calculations (for mean and proportions)
- Bartlett test for equal variances, then Student's or Welch's t-test accordingly
- Chi-square test of independence between gender and alcohol level
- Categorical analysis (low/high alcohol consumption)
- Statistical comparisons and visualization

**Dependencies:** pandas, numpy, matplotlib, seaborn, scipy

**Data Files:** `data/student.csv`

---

### 5. **Matplotlib Visualization Examples** (`matplotlib_visualization_examples.py`)
Comprehensive demonstration of matplotlib plotting techniques and visualization best practices.

**Key Features:**
- Simple line and scatter plots
- Damped cosine wave visualization
- Polynomial subplot examples
- Various plot styles and configurations
- Command-line interface for running examples

**Dependencies:** matplotlib, numpy

**Data Files:** None

---

## Data Files

The `data/` folder includes the following datasets:
- `house_train.csv` - Housing market data for analysis
- `gold_price_data.csv` - Historical gold price data
- `student.csv` - Student demographic and performance data

---

## Directory Structure

```
data_projects/
├── README.md                                  (this file)
├── literary_text_analysis.py
├── housing_market_analysis.py
├── gold_price_prediction.py
├── student_alcohol_correlation.py
├── matplotlib_visualization_examples.py
├── data/
│   ├── house_train.csv
│   ├── gold_price_data.csv
│   └── student.csv
└── outputs/                                   (generated, git-ignored)
    └── <script_name>/                         (plots for each script)
```

---

## Getting Started

### Prerequisites
Install required dependencies from the repository root:
```bash
pip install -r requirements.txt
```

### Running Individual Projects

**Literary Text Analysis:**
```bash
python literary_text_analysis.py
```

**Housing Market Analysis:**
```bash
python housing_market_analysis.py
```

**Gold Price Prediction:**
```bash
python gold_price_prediction.py
```

**Student Alcohol Correlation:**
```bash
python student_alcohol_correlation.py
```

**Matplotlib Examples:**
```bash
python matplotlib_visualization_examples.py --help
python matplotlib_visualization_examples.py
```

---

## Output

Each script saves its plots to `data_projects/outputs/<script_name>/`, except the matplotlib examples, which display their figures interactively. Data files are resolved relative to the script, so the scripts can be run from any working directory.

---

## License

See [LICENSE](../LICENSE) for details.
