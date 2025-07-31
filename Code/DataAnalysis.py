import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler


# =======================
# 1. Load and Prepare Data
# =======================
df = pd.read_csv(r'C:\Users\Jmmen\OneDrive\Desktop\Projects\Breast Cancer Wisconsin.csv')

# Convert diagnosis to numeric: Malignant = 1, Benign = 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Drop non-useful columns
df = df.drop(['id'], axis=1, errors='ignore')


# =======================
# 2. Correlation with Diagnosis - Barplot
# =======================

# Shows top predictive features

corr_with_diagnosis = df.corr()['diagnosis'].sort_values(ascending=False)

plt.figure(figsize=(8, 10))
ax = sns.barplot(x=corr_with_diagnosis.values, y=corr_with_diagnosis.index, palette='coolwarm')
for i, v in enumerate(corr_with_diagnosis.values):
    ax.text(v + 0.01, i, f"{v:.2f}", color='black', va='center')
plt.title("Feature Correlation with Diagnosis", fontsize=16)
plt.xlabel("Correlation Coefficient")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# =======================
# 3. Full Feature Heatmap
# =======================

# Shows overall feature relationships

plt.figure(figsize=(20,18))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False, 
            linewidths=0.5, linecolor='white', cbar=True)
plt.title("Correlation Heatmap", fontsize=18)
plt.show()



# =======================
# 4. Boxplots for Top Features
# =======================

# Shows distribution differences

top3_features = corr_with_diagnosis.index[1:4]  # Skip 'diagnosis'

plt.figure(figsize=(15,5))
for i, feature in enumerate(top3_features, 1):
    plt.subplot(1,3,i)
    sns.boxplot(x='diagnosis', y=feature, data=df, palette=['#2ca02c','#d62728'])
    plt.title(f"{feature} by Diagnosis")
    plt.xlabel("Diagnosis (0=Benign, 1=Malignant)")
plt.tight_layout()
plt.show()


# =======================
# 5. Scatterplots: Mean vs Worst Features
# =======================

# Do tumors with larger "worst" values indicate
#  malignancy more strongly?

mean_worst_pairs = [
    ('radius_mean', 'radius_worst'),
    ('perimeter_mean', 'perimeter_worst'),
    ('concave points_mean', 'concave points_worst')
]

plt.figure(figsize=(15,5))
for i, (mean_f, worst_f) in enumerate(mean_worst_pairs, 1):
    plt.subplot(1,3,i)
    sns.scatterplot(x=df[mean_f], y=df[worst_f], hue=df['diagnosis'], 
                    palette=['#2ca02c','#d62728'], alpha=0.7)
    plt.title(f"{mean_f} vs {worst_f}")
plt.tight_layout()
plt.show()


# =======================
# 6. Severity Index Creation & Visualization
# =======================

# Finding how well top correlated features separates
#  malignant vs. benign cases

scaler = MinMaxScaler()
df_scaled = df.copy()
df_scaled[top3_features] = scaler.fit_transform(df[top3_features])

df['severity_index'] = df_scaled[top3_features].mean(axis=1)

plt.figure(figsize=(6,5))
sns.boxplot(x='diagnosis', y='severity_index', data=df, palette=['#2ca02c','#d62728'])
plt.title("Severity Index by Diagnosis")
plt.xlabel("Diagnosis (0=Benign, 1=Malignant)")
plt.ylabel("Severity Index")
plt.show()