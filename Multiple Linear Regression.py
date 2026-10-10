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
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. Dataset Loading & Exploration
# ==========================================
data = "Dataset/Housing.csv"
df = pd.read_csv(data)
df.head()
df.info()

# ==========================================
# 2. Data Preprocessing & Categorical Encoding
# ==========================================
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# ==========================================
# 3. Train-Test Split & Model Training
# ==========================================
X = df.drop("price", axis=1)
y = df["price"]

x1, x2, y1, y2 = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x1, y1)

# ==========================================
# 4. Model Predictions
# ==========================================
y_pred = model.predict(x2)
y_pred

# ==========================================
# 5. Evaluation Metrics
# ==========================================
mae = mean_absolute_error(y2, y_pred)
mse = mean_squared_error(y2, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y2, y_pred)
mape = mean_absolute_percentage_error(y2, y_pred)
evs = explained_variance_score(y2, y_pred)
maxerr = max_error(y2, y_pred)

n = x2.shape[0]  # Number of observations
p = x2.shape[1]  # Number of predictors

adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

print("\n========== Regression Evaluation ==========")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
print(f"Adjusted R² Score: {adj_r2:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
print(f"Explained Variance Score: {evs:.4f}")
print(f"Maximum Error: {maxerr:.2f}")

# ==========================================
# 6. Model Parameters (Intercept & Coefficients)
# ==========================================
print("\nIntercept:")
print(model.intercept_)

print("\nCoefficients:")
coef = pd.DataFrame({"Feature": X.columns, "Coefficient": model.coef_})
print(coef)

# ==========================================
# 7. Prediction on New Unseen Sample
# ==========================================
new_house = [[5000, 3, 2, 2, 1, 1, 0, 1, 0, 1, 0, 1]]
prediction = model.predict(new_house)

print("\nPredicted House Price:", prediction[0])

# ==========================================
# 8. Visualization (Actual vs Predicted)
# ==========================================
plt.figure(figsize=(6, 6))
plt.scatter(y2, y_pred)
plt.plot([y2.min(), y2.max()], [y2.min(), y2.max()], "r--")

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Values")
plt.show()
