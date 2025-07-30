import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load csv file
df = pd.read_csv(r'C:\Users\Jmmen\OneDrive\Desktop\Projects\Breast Cancer Wisconsin.csv')

# ----------Below is the setup of the barplot---------

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

# ----------------------------------------------

# Boxplot for top features
top_features = corr_with_diagnosis.index[1:6]  # top 5 excluding diagnosis
plt.figure(figsize=(15,8))
for i, feature in enumerate(top_features, 1):
    plt.subplot(2,3,i)
    sns.boxplot(x='diagnosis', y=feature, data=df, palette='Set2')
    plt.title(f"{feature} by Diagnosis")
plt.tight_layout()
plt.savefig("images/top_features_boxplot.png", dpi=300, bbox_inches='tight')
plt.show()

# --------------------------------------------

# Pairplot to show how top features interact between diagnoses

top3_features = corr_with_diagnosis.index[1:4]  # skip diagnosis itself


plt.figure(figsize=(15,5))
for i, feature in enumerate(top3_features, 1):
    plt.subplot(1,3,i)
    sns.stripplot(x='diagnosis', y=feature, data=df, palette=['#2ca02c','#d62728'], alpha=0.6, jitter=True)
    plt.title(f"{feature} by Diagnosis")
    plt.xlabel("Diagnosis (0=Benign, 1=Malignant)")
plt.tight_layout()
plt.savefig("images/pairplot_top_three_features.png", dpi=300, bbox_inches='tight')
plt.show()



