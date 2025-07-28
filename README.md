# Breast Cancer Diagnosis Analysis (Wisconsin Dataset)

This project analyzes the Breast Cancer Wisconsin dataset to explore how different tumor measurements relate to cancer diagnosis (benign vs malignant). The project includes data cleaning, visualization, and a machine learning classification model.

---

## Project Goals

- Understand which features are most correlated with malignant tumors.
- Visualize the relationship between features and diagnosis using heatmaps and boxplots.
- Build a simple classifier to predict diagnosis based on features.

---

## Data Preprocessing

- Dropped ID and unnamed columns.
- Converted `diagnosis` column to binary (`M` → 1, `B` → 0).
- Verified data types and cleaned missing values (none in this dataset).

---

## Exploratory Data Analysis (EDA)

- Plotted diagnosis distribution
- Created feature-by-feature visualizations
- Correlation heatmap shows features like `radius_mean`, `perimeter_mean`, and `concavity_mean` are strongly associated with diagnosis.

---

## Tech Stack

- Python (Pandas, Matplotlib, Seaborn, Scikit-learn)
- VS Code

---

## Files

- `BreastCancerAnalysis.py` – main file with all of the code
- `README.md` – project overview (this file)
- `Breast Cancer Wisconsin.csv` – original dataset
- `images/` – contains saved plots

---

## Sample Visualizations

### Full Correlation Heatmap
Shows the relationships between all of the tumor features:
(images/full_correlation_heatmap.png)

---

## Author

- Jenna M. – Computer Science student

---

## Dataset Source

- [Kaggle - Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/iamtanmayshukla/breast-cancer-diagnostic-data-set/discussion?sort=hotness)