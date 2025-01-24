import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# File paths
pca_file_path = 'c:/Startcoding/your_pca_factors.csv'
pls_file_path = 'c:/Startcoding/your_pls_factors.csv'

# Load PCA data
pca_data = pd.read_csv(pca_file_path)
print("PCA Data:")
print(pca_data.head())
print("PCA Data Shape:", pca_data.shape)

# Load PLS data
pls_data = pd.read_csv(pls_file_path)
print("PLS Data:")
print(pls_data.head())
print("PLS Data Shape:", pls_data.shape)

# Check if the number of rows in PCA and PLS data matches
if pca_data.shape[0] != pls_data.shape[0]:
    raise ValueError("The number of rows in PCA and PLS data does not match.")

# Combine PCA and PLS data
X_pca = pca_data[['PC1', 'PC2']].values
X_pls = pls_data[['Factor1', 'Factor2']].values
X_combined = np.hstack((X_pca, X_pls))

# K-Means clustering
# Use the elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 11):  # Try cluster counts from 1 to 10
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_combined)
    wcss.append(kmeans.inertia_)

# Visualize the elbow method results
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Determine the optimal number of clusters
optimal_k = 3  # Set based on the elbow method results

# Perform K-Means clustering
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_combined)

# Add cluster labels to the PCA data
pca_data['Cluster'] = clusters

# Extract class labels
y = pca_data['Cluster'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Train K-NN classifier
knn = KNeighborsClassifier(n_neighbors=15)  # Set k=15
knn.fit(X_train, y_train)

# Make predictions on test data
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"K-NN Classification Accuracy: {accuracy:.4f}")

# Visualize K-Means clustering results (2D)
plt.figure(figsize=(10, 6))
plt.scatter(X_combined[:, 0], X_combined[:, 1], c=clusters, cmap='viridis', marker='o')
plt.title('K-Means Clustering Results (2D)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# Visualize K-NN classification results (2D)
plt.figure(figsize=(10, 6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', marker='x', label='Predicted')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', marker='o', alpha=0.5, label='Training')
plt.title('K-NN Classification Results (2D)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Class')
plt.legend()
plt.grid(True)
plt.show()
