# Python Projects Repository

## Overview
This repository contains **personal and learning projects** focused on Python programming, data science, and automation. These projects are based on research from textbooks, online resources, and practical experimentation.

## Contents
- [`data_projects/`](data_projects/README.md) — data analysis, statistics, and machine learning scripts
- `payment_receipt.py` — generates a PDF payment receipt with ReportLab

## Technologies
- Python 3.10+ (pandas, NumPy, SciPy, scikit-learn, XGBoost)
- Matplotlib / Seaborn
- ReportLab

## Purpose
To practice Python concepts and apply them to real-world scenarios, reinforcing skills in data manipulation, analytics, and automation.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/Emmanuel-Esplin/Python_Projects.git
   cd Python_Projects
   ```
2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run any script directly, for example:
   ```bash
   python data_projects/housing_market_analysis.py
   python payment_receipt.py
   ```

Scripts locate their data files relative to their own location, so they can be run from any directory. Generated plots and PDFs are written to an `outputs/` folder next to each script (ignored by git).

## GitHub Profile
For more projects: https://github.com/Emmanuel-Esplin/Python_Projects
