import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==========================================
# 1. Dataset Loading & Exploration
# ==========================================
df = pd.read_csv("heart_disease_dataset.csv")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTarget Distribution:")
print(df["Heart Disease"].value_counts())

# ==========================================
# 2. Feature Selection & Identification
# ==========================================
X = df.drop("Heart Disease", axis=1)
y = df["Heart Disease"]

numerical_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

print("Numerical Features:")
print(list(numerical_features))

print("\nCategorical Features:")
print(list(categorical_features))

# ==========================================
# 3. Train-Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Training Data:", X_train.shape)
print("Testing Data :", X_test.shape)

# ==========================================
# 4. Data Preprocessing & Pipeline
# ==========================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ==========================================
# 5. Baseline KNN Model (K = 5)
# ==========================================
K = 5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_processed, y_train)

y_pred = knn.predict(X_test_processed)

print("Predicted Values:")
print(y_pred[:20])

print(classification_report(y_test, y_pred))

# ==========================================
# 6. Hyperparameter Tuning (Searching Best K)
# ==========================================
k_values = [1, 3, 5, 7, 9, 11, 15, 21]

accuracy_values = []
precision_values = []
recall_values = []
f1_values = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_processed, y_train)
    y_pred = knn.predict(X_test_processed)

    accuracy_values.append(accuracy_score(y_test, y_pred))
    precision_values.append(precision_score(y_test, y_pred))
    recall_values.append(recall_score(y_test, y_pred))
    f1_values.append(f1_score(y_test, y_pred))

results = pd.DataFrame(
    {
        "K": k_values,
        "Accuracy": accuracy_values,
        "Precision": precision_values,
        "Recall": recall_values,
        "F1 Score": f1_values,
    }
)

print(results)

# ==========================================
# 7. K-Value Visualization
# ==========================================
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracy_values, marker="o")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("Effect of K Value on KNN Accuracy")
plt.xticks(k_values)
plt.grid()
plt.show()

# ==========================================
# 8. Optimal K Metric Output
# ==========================================
best_index = np.argmax(f1_values)
best_k = k_values[best_index]

print("Best K:", best_k)
print("Best Accuracy:", accuracy_values[best_index])
print("Best Precision:", precision_values[best_index])
print("Best Recall:", recall_values[best_index])
print("Best F1 Score:", f1_values[best_index])

# ==========================================
