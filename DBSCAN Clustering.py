import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==========================================
# 1. Dataset Loading & Inspection
# ==========================================
df = pd.read_csv("heart_disease_dataset.csv")

print("Dataset Shape:", df.shape)

print("\nFirst 5 records:")
print(df.head())

# ==========================================
# 2. Feature Selection & Categorization
# ==========================================
X = df.drop("Heart Disease", axis=1)

numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

# ==========================================
# 3. Preprocessing Setup & Transformation
# ==========================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

X_processed = preprocessor.fit_transform(X)
X_processed = X_processed

# ==========================================
# 4. DBSCAN Clustering Model
# ==========================================
dbscan = DBSCAN(eps=1.5, min_samples=5)

labels = dbscan.fit_predict(X_processed)

df["Cluster"] = labels

# ==========================================
# 5. Cluster & Noise Analysis
# ==========================================
print("\nCluster Labels:")
print(df["Cluster"].value_counts().sort_index())

noise_points = df[df["Cluster"] == -1]

print("\nNumber of Noise Points:")
print(len(noise_points))

print("\nNoise Points:")
print(noise_points.head(10))

number_of_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print("\nNumber of Clusters:")
print(number_of_clusters)

# ==========================================
# 6. Dimensionality Reduction (PCA)
# ==========================================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

# ==========================================
# 7. Cluster Visualization
# ==========================================
plt.figure(figsize=(9, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=labels,
    cmap="viridis",
    s=40,
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("DBSCAN Clustering")

plt.colorbar(label="Cluster")

plt.show()
