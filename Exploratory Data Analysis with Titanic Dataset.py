import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# 1. Dataset Loading
# ==========================================
data = "Dataset/titanic.csv"
df = pd.read_csv(data)

# ==========================================
# 2. Exploratory Data Analysis (EDA)
# ==========================================
df.head()
df.tail()
df.sample(5)
print("Rows and Columns:", df.shape)
df.info()
df.describe()
df.size
df.columns

# ==========================================
# 3. Missing Value Analysis & Visualization
# ==========================================
df.isnull().sum()
missing_percentage = (df.isnull().sum() / len(df)) * 100
print(missing_percentage.sort_values(ascending=False))

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

# ==========================================
# 4. Data Cleaning & Imputation
# ==========================================
# Fill Age missing values
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill Embarked missing values
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Fill Cabin missing values
df["Cabin"] = df["Cabin"].fillna("Unknown")

print(df.isnull().sum())

# ==========================================
# 5. Duplicate Removal
# ==========================================
print("Number of duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()
print("Shape after removing duplicates:", df.shape)

# ==========================================
# 6. Univariate Visualization
# ==========================================
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Survived")
plt.title("Passenger Survival Count")
plt.xlabel("Survived")
plt.ylabel("Number of Passengers")
plt.show()

survival_counts = df["Survived"].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(
    survival_counts,
    labels=["Did Not Survive", "Survived"],
    autopct="%1.1f%%",
    startangle=90,
)
plt.title("Survival Distribution")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Pclass")
plt.title("Passenger Class Distribution")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Sex")
plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.show()

# ==========================================
# 7. Bivariate & Multivariate Analysis
# ==========================================
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Sex", hue="Survived")
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.show()

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Pclass", hue="Survived")
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.show()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="Survived", y="Fare")
plt.title("Fare Distribution by Survival")
plt.xlabel("Survived")
plt.ylabel("Fare")
plt.show()

g = sns.catplot(
    data=df,
    x="Pclass",
    hue="Survived",
    col="Sex",
    kind="count",
    height=5,
    aspect=1,
)
g.fig.suptitle("Survival by Passenger Class and Gender", y=1.05)
plt.show()

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df,
    x="Age",
    y="Fare",
    hue="Survived",
    style="Sex",
    size="Pclass",
    sizes=(50, 200),
)
plt.title("Age vs Fare by Survival, Gender and Class")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()

# ==========================================
# 8. Correlation Analysis
# ==========================================
numeric_columns = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
correlation = df[numeric_columns].corr()
print(correlation)

plt.figure(figsize=(9, 7))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()
