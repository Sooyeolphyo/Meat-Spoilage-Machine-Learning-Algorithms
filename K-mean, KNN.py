import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 파일 경로 설정
pca_file_path = 'c:/Startcoding/your_pca_factors2.csv'
pls_file_path = 'c:/Startcoding/your_pls_factors2.csv'

# PCA 데이터 로드
pca_data = pd.read_csv(pca_file_path)
print("PCA Data:")
print(pca_data.head())
print("PCA Data Shape:", pca_data.shape)

# PLS 데이터 로드
pls_data = pd.read_csv(pls_file_path)
print("PLS Data:")
print(pls_data.head())
print("PLS Data Shape:", pls_data.shape)

# PCA와 PLS 데이터의 행 수 확인
if pca_data.shape[0] != pls_data.shape[0]:
    raise ValueError("PCA와 PLS 데이터의 행 수가 일치하지 않습니다.")

# PCA 데이터와 PLS 데이터 결합
X_pca = pca_data[['PC1', 'PC2']].values
X_pls = pls_data[['Factor2', 'Factor3']].values
X_combined = np.hstack((X_pca, X_pls))

# 클러스터링을 위한 K-Means
# 엘보우 방법을 사용하여 최적의 클러스터 수 찾기
wcss = []
for i in range(1, 11):  # 1에서 10까지의 클러스터 수 시도
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(X_combined)
    wcss.append(kmeans.inertia_)

# 엘보우 방법 결과 시각화
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# 최적의 클러스터 수 결정
optimal_k = 3  # 엘보우 방법 결과를 기반으로 설정한 값으로 수정

# K-Means 클러스터링 수행
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
clusters = kmeans.fit_predict(X_combined)

# 클러스터 레이블을 PCA 데이터에 추가
pca_data['Cluster'] = clusters

# 클래스 레이블 추출
y = pca_data['Cluster'].values

# 학습 데이터와 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# K-NN 분류기 학습
knn = KNeighborsClassifier(n_neighbors=15)  # k=5로 설정
knn.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
y_pred = knn.predict(X_test)

# 정확도 계산
accuracy = accuracy_score(y_test, y_pred)
print(f"K-NN Classification Accuracy: {accuracy:.4f}")

# K-Means 클러스터링 결과 시각화 (2D)
plt.figure(figsize=(10, 6))
plt.scatter(X_combined[:, 0], X_combined[:, 1], c=clusters, cmap='viridis', marker='o')
plt.title('K-Means Clustering Results (2D)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()

# K-NN 분류 결과 시각화 (2D)
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
