## Author

- Jenna M. – Computer Science student

---

# Breast Cancer Diagnosis Analysis (Wisconsin Dataset)

This project explores the Breast Cancer Wisconsin dataset using Python, Pandas, Seaborn, and Matplotlib.  
The analysis identifies which tumor features most strongly correlate with malignancy, visualizes feature relationships, and introduces a custom **Severity Index** to assess tumor risk.

---

## Project Goals

- Understand which features are most correlated with malignant tumors.
- Visualize the relationship between features and diagnosis using different visualizations
- Build a simple classifier to predict diagnosis based on features.

---

## Files

- `BreastCancerAnalysis.py` – main file with all of the code for the visualizations
- `Predict_Tumor.py` - code for the prediction model
- `README.md` – project overview (this file)
- `Breast Cancer Wisconsin.csv` – original dataset
- `images/` – contains the visualizations

---

## Data Preprocessing

- Dropped ID and unnamed columns.
- Converted `diagnosis` column to binary (`M` → 1, `B` → 0).
- Verified data types and ensured that there is no missing values (none in this dataset).

---

### Features

- Correlation analysis of features with diagnosis
- Clustered heatmap showing relationships between features
- Boxplots comparing top predictive features by diagnosis
- Scatterplots comparing mean vs. worst tumor measurements
- Custom **Severity Index** based on top features
- Prediction model which shows the likihood of a tumor being malignant and benign

### Key Findings

- Can be found on the wiki page on Github:
        https://github.com/menajm/DataAnalysisProject1/wiki

---

## Dataset Source

- [Kaggle - Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/iamtanmayshukla/breast-cancer-diagnostic-data-set/discussion?sort=hotness)