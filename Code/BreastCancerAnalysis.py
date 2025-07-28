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

#This graph will show how ALL features relate to each other
plt.figure(figsize = (20, 18)) # Customize the size
sns.heatmap(df.corr(), cmap = 'coolwarm', annot = False, linewidths = 0.5)
plt.title("Correlation Heatmap", fontsize = 18)
plt.show()