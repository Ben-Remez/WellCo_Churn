
# WellCo Member Churn Prediction
This project was developed as part of the Vi Data Science Home Assignment.<br>
Its goal is to identify WellCo members at highest risk of churn and provide an actionable outreach list to guide retention efforts.

The work combines data exploration, feature engineering, predictive modeling, evaluation, and operational optimization - following an end-to-end data science pipeline.

---

## Project Overview
- **Objective:** Predict member churn likelihood and prioritize outreach candidates.
- **Business Impact:** By targeting the top 5–10% most at-risk members, WellCo can double the efficiency of retention campaigns relative to baseline.
- **Approach:**
    - Build interpretable, robust models using control-group data only (to avoid treatment leakage).
    - Evaluate performance on a hold-out set (ROC-AUC, PR-AUC, Precision@k).
    - Generate a ranked list of members by predicted churn risk for operational use.

---

## Repository Structure
Wellco_churn/<br>
│<br>
├── 01_data_preperation.ipynb<br>
│  └─ Cleans, explores, and engineers features from app, web, and claims data.<br>
│<br>
├── 02_modeling_and_evaluation.ipynb<br>
│  └─ Trains, evaluates, and interprets models; produces outreach ranking.<br>
│<br>
├── src/<br>
│ ├── helpers.py # General purpose data helpers<br>
│ ├── eval.py # Evaluation metrics (ROC/PR, precision@k, calibration)<br>
│ ├── visualization.py # Basic Visualizations<br>
│<br>
├── artifacts/<br>
│ ├── features_merged.parquet<br>
│ ├── train_control.parquet<br>
│ ├── holdout_control.parquet<br>
│<br>
├── data.zip/<br> # Contains the relevant data for this project.
│<br>
├── results/<br>
│ ├── top_n_members.csv # final ranked outreach list<br>
├── stakeholders_presentation.pdf # 3–5 slide executive summary.<br>
├── requirements.txt<br>
└── README.md

---

## Setup & Reproducibility
You can run the notebooks in any environment that supports Jupyter, but the project was tested primarily in Visual Studio Code with the Python extension.<br>
Make sure to extract the data from the data.zip file as a data directory containing the relevant files. Extract the data directory under the project root directory.

**Python Version & Environment**<br>
This project was created using Python version 3.10.4 and Visual Studio Code.<br>

If you decide to use Visual Studio Code to run this project, please perform the following steps to set up a virtual environment (if you dont already have one set up).

**Creating a Virtual Environment**<br>
Open a terminal in the project folder and run:
```bash
# Create Virtual Environment.
python -m venv .venv

# Activate it
# (Windows)
.venv\Scripts\activate
# (macOS / Linux)
source .venv/bin/activate
```

**Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Run Order**
1. Execute `01_data_preperation.ipynb` - Generated cleaned parquet files in `artifacts/`.
2. Execute `02_modeling_and_evaluation.ipynb` - Trains models, evaluates performance, and outputs `top_n_members.csv`.

All notebooks are reproducible top-to-bottom and require no external credentials.

---

## Modeling Summary
| Model                | ROC-AUC | PR-AUC | Key Traits                                |
| :------------------- | :------ | :----- | :---------------------------------------- |
| Logistic Regression  | ~0.66   | ~0.30  | Linear, interpretable baseline            |
| HistGradientBoosting | ~0.64   | ~0.30  | Non-linear, captures feature interactions |

- **Top Features:** `web_visits`, `web_unique_titles`, `other` (different content than main categories), `tenure_days`, `app_sessions`, `app_activity_days`, `claims_count`.
- **Calibration:** Mild over-confidence corrected via isotonic calibration.
- **Operational Metric:** Precision@1% ≈ 0.45 (≈2× lift over baseline churn rate ~20%).

---

## Operational Insights
- **Optimal Target Fraction:** Top 5–10% of members - High precision (≈0.35–0.4) while capturing ~25% of churners.
- **Final Deliverable:** `results/top_n_members.csv`.
    - Columns: `member_id`, `score`, `rank`.
    - Sorted descending by churn probability.
- **Interpretation:** This represents WellCo’s most at-risk members for immediate outreach.

---

## Key Learnings & Next Steps
- Engagement and tenure dominate churn prediction.
- Behavioral features (web/app usage, topic shares) outperform demographic or diagnostic ones.
- Recommended next steps (For a real world application):
    1. Hyperparameter optimization and calibration refinement.
    2. Deploy the model to score members weekly or monthly.
    3. Integrate feedback loops to measure retention campaign impact.

---
## Stakeholders Presentation
A short presentation was added, transfering the insights for non-technical stakeholders.<br>

You can find this presentation in the hope repository page.

---

**Author:** *Ben Remez*<br>
**Date:** October 2025<br>
**Language:** Python (3.10)<br>