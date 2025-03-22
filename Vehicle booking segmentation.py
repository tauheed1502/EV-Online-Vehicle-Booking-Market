# vehicle_booking_segmentation.py
# This script applies Decision Trees and Random Forest for segmenting the online vehicle booking market

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset
df = pd.read_csv("vehicle_booking_data.csv")

# Display basic info
df.info()
print("\n", df.head())

# Handle missing values
df.fillna(df.mode().iloc[0], inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_cols = ['City', 'User_Type', 'Preferred_Payment_Mode']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Feature selection
features = ['City', 'Income', 'Usage_Frequency', 'User_Type', 'Preferred_Payment_Mode']
target = 'Segment'

X = df[features]
y = df[target]

# Normalize numerical features
scaler = StandardScaler()
X[['Income', 'Usage_Frequency']] = scaler.fit_transform(X[['Income', 'Usage_Frequency']])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree model
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_dt = dt_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Evaluation
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report (Random Forest):\n", classification_report(y_test, y_pred_rf))

# Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Save results
df['Predicted_Segment'] = rf_model.predict(X)
df.to_csv("vehicle_booking_segmented.csv", index=False)

# Visualizing Decision Tree
plt.figure(figsize=(15,8))
plot_tree(dt_model, feature_names=features, class_names=[str(c) for c in df[target].unique()], filled=True)
plt.title("Decision Tree Visualization")
plt.show()

print("Vehicle booking market segmentation complete. Results saved as vehicle_booking_segmented.csv")
