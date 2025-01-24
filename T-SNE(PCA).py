import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# Load your data
df = pd.read_csv('c:/Startcoding/your_pca_factors.csv')

# Assuming the first columns contain the data and the last column contains the labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Convert labels to numeric format using LabelEncoder
label_encoder = LabelEncoder()
y_numeric = label_encoder.fit_transform(y)

# Create a mapping from numeric labels to sample names
label_mapping = dict(zip(y_numeric, y))

# Apply PCA on the data to get the first two principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Apply T-SNE on the PCA components
tsne = TSNE(n_components=2, perplexity=5, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_pca)

# Plotting the T-SNE result
plt.figure(figsize=(14, 10))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_numeric, cmap='viridis', alpha=0.6)

# Add labels to each point
for i, txt in enumerate(y):
    plt.text(X_tsne[i, 0], X_tsne[i, 1], txt, fontsize=9, ha='right')

# Create a colorbar with both encoded labels and sample names
cbar = plt.colorbar(scatter, ticks=np.arange(len(label_mapping)))
cbar.set_label('Encoded Labels')
cbar.set_ticks(list(label_mapping.keys()))
cbar.set_ticklabels([f"{key} - {value}" for key, value in label_mapping.items()])

# Adjust the font size of the colorbar labels
cbar.ax.tick_params(labelsize=8)  # Reduce font size here

plt.title('T-SNE Visualization of PCA Scores with Sample Names')
plt.xlabel('T-SNE Component 1')
plt.ylabel('T-SNE Component 2')
plt.grid(True)
plt.show()
