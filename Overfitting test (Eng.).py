import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, learning_curve, train_test_split
import matplotlib.pyplot as plt

# Loading and combining PCA and PLS data (same as previous code)
pca_data = pd.read_csv('c:/Startcoding/your_pca_factors.csv')
pls_data = pd.read_csv('c:/Startcoding/your_pls_factors.csv')

X_pca = pca_data[['PC1', 'PC2']].values
X_pls = pls_data[['Factor1', 'Factor2']].values
X_combined = np.hstack((X_pca, X_pls))

# Performing K-Means clustering (same as previous code)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_combined)
pca_data['Cluster'] = clusters
y = pca_data['Cluster'].values

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# 1. Checking for overfitting using cross-validation
knn = KNeighborsClassifier(n_neighbors=15)
cv_scores = cross_val_score(knn, X_combined, y, cv=5)  # 5-Fold cross-validation
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Average CV Score: {np.mean(cv_scores):.4f}")

# 2. Visualizing the learning curve
train_sizes, train_scores, test_scores = learning_curve(knn, X_combined, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10))

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
plt.plot(train_sizes, test_mean, 'o-', color='red', label='Cross-validation score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.2)
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color='red', alpha=0.2)
plt.title('Learning Curve')
plt.xlabel('Training Size')
plt.ylabel('Score')
plt.legend(loc='best')
plt.grid(True)
plt.show()
