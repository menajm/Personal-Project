import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load csv file
df = pd.read_csv(r'C:\Users\Jmmen\OneDrive\Desktop\Projects\Breast Cancer Wisconsin.csv')

# ----------Below is the setup of the Heatmap---------

# Convert the diagnosis to numeric: Malignant = 1
# Benign = 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

#Drop the columns that aren't useful for correlation
df = df.drop(['id'], axis = 1, errors = 'ignore')

# --------------------------------------------------

# Calculate correlation of all features with the diagnosis column
corr_with_diagnosis = df.corr()['diagnosis'].sort_values(ascending=False)

# Plot the correlation as a barplot
plt.figure(figsize=(8, 10))
sns.barplot(x=corr_with_diagnosis.values, y=corr_with_diagnosis.index, palette='coolwarm')
plt.title("Feature Correlation with Diagnosis", fontsize=16)
plt.xlabel("Correlation Coefficient")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("images/diagnosis_correlation_barplot.png", dpi=300, bbox_inches='tight')
plt.show()
