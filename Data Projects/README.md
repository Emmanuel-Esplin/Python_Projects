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

**Data Files:** `house_train.csv`

---

### 3. **Gold Price Prediction** (`gold_price_prediction.py`)
Machine learning project predicting gold prices using various regression models.

**Key Features:**
- Loads and explores gold price dataset
- Correlation analysis with heatmap visualization
- Feature preprocessing (removes highly correlated features)
- Implements multiple ML models:
  - Polynomial Regression (Lasso)
  - Random Forest Regressor
  - XGBoost Regressor
- Model evaluation using R² and RMSE metrics
- Grid search for hyperparameter tuning

**Dependencies:** pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost

**Data Files:** `gold_price_data.csv`

---

### 4. **Student Alcohol Correlation Analysis** (`student_alcohol_correlation.py`)
Statistical analysis exploring the relationship between student alcohol consumption and academic performance.

**Key Features:**
- Loads and cleans student data
- Computes alcohol consumption index
- Confidence interval calculations (for mean and proportions)
- Statistical hypothesis testing
- Categorical analysis (low/high alcohol consumption)
- Statistical comparisons and visualization

**Dependencies:** pandas, numpy, matplotlib, seaborn, scipy

**Data Files:** `student.csv`

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

The `Data Projects` folder includes the following datasets:
- `house_train.csv` - Housing market data for analysis
- `gold_price_data.csv` - Historical gold price data
- `student.csv` - Student demographic and performance data

---

## Directory Structure

```
Data Projects/
├── README.md                                  (this file)
├── literary_text_analysis.py
├── housing_market_analysis.py
├── gold_price_prediction.py
├── student_alcohol_correlation.py
├── matplotlib_visualization_examples.py
├── house_train.csv
├── gold_price_data.csv
├── student.csv
└── outputs/
    └── plots/                                 (generated visualizations)
```

---

## Getting Started

### Prerequisites
Install required dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy requests datascience
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

## Features & Best Practices

All projects include:
- ✅ Proper logging and error handling
- ✅ Type hints for better code clarity
- ✅ Modular function design
- ✅ Data validation and preprocessing
- ✅ Professional visualization styling
- ✅ Output directory management

---

## Output

Generated visualizations and analysis results are saved to the `outputs/plots/` directory.

---

## Author

Emmanuel Esplin

---

## License

See [LICENSE](../LICENSE) for details.
