# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Personal/learning collection of standalone Python scripts (data analysis, ML, visualization, small utilities). There is no package, build system, test suite, or linter config. Each script is independent and run directly.

## Setup and running

```bash
pip install -r requirements.txt
python data_projects/housing_market_analysis.py
python data_projects/student_alcohol_correlation.py
python data_projects/gold_price_prediction.py          # grid searches take ~1 min
python data_projects/literary_text_analysis.py         # fetches book texts over the network
python data_projects/matplotlib_visualization_examples.py -e {all,simple,damped,poly,trig,olympics,hist}
python payment_receipt.py
```

To run headless (no plot windows), prefix with `MPLBACKEND=Agg`. `requirements.txt` pins `setuptools<81` because `datascience` still imports `pkg_resources`.

## Conventions

- Every script follows the same shape: typed functions, `logging`, a `main()` behind `if __name__ == "__main__"`. Keep new scripts consistent with this.
- Paths are resolved from `Path(__file__).parent`, never the CWD or absolute paths. Datasets (`*.csv`) live in `data_projects/data/`; generated files go to `outputs/<script_name>/` next to the script, which is git-ignored.
- `gold_price_prediction.py` targets `GLD` (gold), not `EUR/USD`, and deliberately uses a chronological split, `TimeSeriesSplit` CV, and train-only outlier limits to avoid look-ahead leakage. Its models currently underperform the naive baseline it prints — that is an honest result, not a bug to "fix" by reintroducing a random split.
