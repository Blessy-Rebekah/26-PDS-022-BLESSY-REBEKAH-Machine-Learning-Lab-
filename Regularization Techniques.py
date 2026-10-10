import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# ==========================================
# 1. Dataset Loading & Encoding
# ==========================================
data = "Dataset/Housing.csv"
df = pd.read_csv(data)
df.head()

df = pd.get_dummies(df, drop_first=True)
df.head()

# ==========================================
# 2. Feature Selection & Train-Test Split
# ==========================================
X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==========================================
# 3. Feature Scaling & Polynomial Features
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

poly = PolynomialFeatures(degree=4, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

print("Original Features :", X_train.shape[1])
print("Polynomial Features :", X_train_poly.shape[1])

# ==========================================
# 4. Linear Regression
# ==========================================
linear = LinearRegression()
linear.fit(X_train_poly, y_train)

train_pred = linear.predict(X_train_poly)
test_pred = linear.predict(X_test_poly)

train_r2 = r2_score(y_train, train_pred)
test_r2 = r2_score(y_test, test_pred)

print("LINEAR REGRESSION")
print("Training R2 :", round(train_r2, 4))
print("Testing R2 :", round(test_r2, 4))

# ==========================================
# 5. Ridge Regression
# ==========================================
ridge = Ridge(alpha=100)
ridge.fit(X_train_poly, y_train)

train_pred = ridge.predict(X_train_poly)
test_pred = ridge.predict(X_test_poly)

ridge_train = r2_score(y_train, train_pred)
ridge_test = r2_score(y_test, test_pred)

print("RIDGE")
print("Training R2 :", round(ridge_train, 4))
print("Testing R2 :", round(ridge_test, 4))

# ==========================================
# 6. Lasso Regression
# ==========================================
lasso = Lasso(alpha=0.1, max_iter=50000)
lasso.fit(X_train_poly, y_train)

train_pred = lasso.predict(X_train_poly)
test_pred = lasso.predict(X_test_poly)

lasso_train = r2_score(y_train, train_pred)
lasso_test = r2_score(y_test, test_pred)

print("LASSO")
print("Training R2 :", round(lasso_train, 4))
print("Testing R2 :", round(lasso_test, 4))

# ==========================================
# 7. ElasticNet Regression
# ==========================================
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=50000)
elastic.fit(X_train_poly, y_train)

train_pred = elastic.predict(X_train_poly)
test_pred = elastic.predict(X_test_poly)

elastic_train = r2_score(y_train, train_pred)
elastic_test = r2_score(y_test, test_pred)

print("ELASTIC NET")
print("Training R2 :", round(elastic_train, 4))
print("Testing R2 :", round(elastic_test, 4))

# ==========================================
# 8. Model Comparison &
