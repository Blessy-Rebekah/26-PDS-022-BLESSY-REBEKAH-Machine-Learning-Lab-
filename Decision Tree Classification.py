import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ==========================================
# 1. Dataset Loading & Inspection
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
# 4. Preprocessing & Model Pipeline Setup
# ==========================================
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns,
        )
    ],
    remainder="passthrough",
)

decision_tree = DecisionTreeClassifier(
    criterion="gini", max_depth=4, random_state=42
)

model = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", decision_tree)]
)

# ==========================================
# 5. Train-Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# ==========================================
# 6. Model Training
# ==========================================
model.fit(X_train, y_train)

print("\nDecision Tree Model Trained Successfully!")

# ==========================================
# 7. Predictions & Evaluation Metrics
# ==========================================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print("\n==========================================")
print("        DECISION TREE EVALUATION")
print("==========================================")

print("Accuracy  :", round(accuracy, 4))
print("Precision :", round(precision, 4))
print("Recall    :", round(recall, 4))
print("F1 Score  :", round(f1, 4))

# ==========================================
# 8. Classification Report
# ==========================================
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==========================================
# 9. Confusion Matrix & Display
# ==========================================
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["No Heart Disease", "Heart Disease"]
)

disp.plot()
plt.title("Confusion Matrix - Decision Tree")
plt.show()

# ==========================================
# 10. Decision Tree Structure Visualization
# ==========================================
# Get the trained classifier
tree_model = model.named_steps["classifier"]

# Get feature names after One-Hot Encoding
feature_names = model.named_steps["preprocessor"].get_feature_names_out()

plt.figure(figsize=(25, 12))

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=["No Heart Disease", "Heart Disease"],
    filled=True,
    rounded=True,
    fontsize=8,
)

plt.title("Decision Tree for Heart Disease Classification")
plt.show()
