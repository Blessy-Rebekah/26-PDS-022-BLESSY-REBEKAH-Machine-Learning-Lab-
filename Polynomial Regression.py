import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

# ==========================================
# 1. Dataset Loading & Inspection
# ==========================================
data = "Dataset/fish.csv"
df = pd.read_csv(data)
df.head()
df.info()
df.shape

# ==========================================
# 2. Preprocessing & One-Hot Encoding
# ==========================================
df = pd.get_dummies(df, columns=["Species"], drop_first=True)

X = df.drop("Weight", axis=1)
y = df["Weight"]

# ==========================================
# 3. Train-Test Split
# ==========================================
x1, x2, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 4. Polynomial Feature Transformation & Training
# ==========================================
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(x1)
X_test_poly = poly.transform(x2)

model = LinearRegression()
model.fit(X_train_poly, y_train)

# ==========================================
# 5. Prediction & Evaluation Metrics
# ==========================================
y_pred = model.predict(X_test_poly)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
maxerr = max_error(y_test, y_pred)

# Adjusted R² Calculation
n = X_test_poly.shape[0]
p = X_test_poly.shape[1] - 1

adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

print("\n========== Evaluation Metrics ==========")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE :", rmse)
print("R2 Score :", r2)
print("Adjusted R2 :", adj_r2)
print("MAPE :", mape)
print("Explained Variance :", evs)
print("Maximum Error :", maxerr)

# ==========================================
# 6. Single Feature Polynomial Visualization
# ==========================================
X = df[["Length3"]]
y = df["Weight"]

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

X_grid = np.arange(X.min().values[0], X.max().values[0], 0.1).reshape(-1, 1)

plt.figure(figsize=(8, 6))
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(
    X_grid,
    model.predict(poly.transform(X_grid)),
    color="red",
    linewidth=3,
    label="Polynomial Curve",
)
plt.xlabel("Length3")
plt.ylabel("Weight")
plt.title("Polynomial Regression")
plt.legend()
plt.show()
