import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ==========================================
# 1. Dataset Loading & Inspection
# ==========================================
data = "Dataset/Salary.csv"
df = pd.read_csv(data)
df = df[["YearsExperience", "Salary"]]
df
df.size
df.shape
df.isnull().sum()

# ==========================================
# 2. Feature Selection & Train-Test Split
# ==========================================
X = df[["YearsExperience"]]  # Independent Variable
y = df["Salary"]  # Dependent Variable

model = LinearRegression()
model.fit(x1, y1)

x1, x2, y1, y2 = train_test_split(X, y, random_state=42, test_size=0.2)

# ==========================================
# 3. Model Prediction
# ==========================================
y_pred = model.predict(x2)

print("Actual Values:")
print(y2.values)
print("Predicted Values:")
print(y_pred)

# ==========================================
# 4. Model Evaluation Metrics
# ==========================================
print("Coefficient (Slope):", model.coef_[0])
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# ==========================================
# 5. Prediction on New Data
# ==========================================
YearsExperience = [[2.1]]
predicted_salary = model.predict(YearsExperience)
print("Predicted salary for 2.1 years:", predicted_salary[0])

# ==========================================
# 6. Visualization
# ==========================================
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(
    X, model.predict(X), color="red", linewidth=2, label="Regression Line"
)
plt.xlabel("Experience")
plt.ylabel("salary")
plt.title("Simple Linear Regression")
plt.legend()
plt.show()
