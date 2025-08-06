import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# =====================
# 1. Load & Prepare Data
# =====================
df = pd.read_csv("Breast Cancer Wisconsin.csv")
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
df = df.drop(['id'], axis=1, errors='ignore')

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Scale features for logistic regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split for training/testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# =====================
# 2. Train Model
# =====================
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# =====================
# 3. Check Accuracy
# =====================
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# =====================
# 3. Ask User for Tumor Measurements
# =====================
print("Enter tumor measurements as prompted:\n")

# We'll ask for only the top 3 features
features = ['concave points_worst', 'perimeter_worst', 'concave points_mean']
user_input = []

for feature in features:
    value = float(input(f"Enter {feature}: "))
    user_input.append(value)

# For missing features, use average values
default_values = X.mean().values
full_input = default_values.copy()

# Replace the top 3 with user-provided values
for i, feature in enumerate(X.columns):
    if feature in features:
        full_input[list(X.columns).index(feature)] = user_input[features.index(feature)]

# =====================
# 4. Scale and Predict
# =====================
sample_scaled = scaler.transform([full_input])
prob = model.predict_proba(sample_scaled)[0]

# =====================
# 5. Display Results
# =====================
print("\n Prediction Results:")
print(f"Probability Benign: {prob[0]*100:.2f}%")
print(f"Probability Malignant: {prob[1]*100:.2f}%")

if prob[1] > 0.5:
    print("The tumor is more likely to be MALIGNANT.")
else:
    print("The tumor is more likely to be BENIGN.")