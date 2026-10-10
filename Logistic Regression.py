import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# ==========================================
# 1. Dataset Loading
# ==========================================
df = pd.read_csv("heart_disease_dataset.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

# ==========================================
# 2. Features & Target Selection
# ==========================================
X = df.drop("Heart Disease", axis=1)
y = df["Heart Disease"]

print("\nFeatures:")
print(X.columns)

print("\nTarget:")
print(y.name)

# ==========================================
# 3. Categorical & Numerical Identification
# ==========================================
categorical_columns = X.select_dtypes(include=["object"]).columns
numerical_columns = X.select_dtypes(exclude=["object"]).columns

print("\nCategorical Columns:")
print(list(categorical_columns))

print("\nNumerical Columns:")
print(list(numerical_columns))

# ==========================================
# 4. Data Imputation
# ==========================================
imputer = SimpleImputer(strategy="most_frequent")
X[categorical_columns] = imputer.fit_transform(X[categorical_columns])

print("\nMissing values after imputation:")
print(X.isnull().sum())

# ==========================================
# 5. Label Encoding
# ==========================================
label_encoders = {}

for column in categorical_columns:
    encoder = LabelEncoder()
    X[column] = encoder.fit_transform(X[column])
    label_encoders[column] = encoder

print("\nData after Label Encoding:")
print(X.head())

# ==========================================
# 6. Train-Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data :", X_test.shape)

# ==========================================
# 7. Feature Scaling
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# minmax_scaler = MinMaxScaler()
# X_train_scaled = minmax_scaler.fit_transform(X_train)
# X_test_scaled = minmax_scaler.transform(X_test)

# ==========================================
# 8. Model Training
# ==========================================
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

print("\nLogistic Regression model trained successfully.")

# ==========================================
# 9. Model Predictions
# ==========================================
y_pred = model.predict(X_test_scaled)
# Probability of class 1
y_probability = model.predict_proba(X_test_scaled)[:, 1]

# ==========================================
# 10. Evaluation Metrics
# ==========================================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n================================")
print("CLASSIFICATION METRICS")
print("==============================
