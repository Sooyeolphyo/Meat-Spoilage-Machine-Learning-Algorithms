import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

# Load your PCA and PLS data
# Assuming 'df_pca' contains PC1 and PC2, and 'df_pls' contains Factor1 and Factor2
df_pca = pd.read_csv('c:/Startcoding/your_pca_factors.csv')
df_pls = pd.read_csv('c:/Startcoding/your_pls_factors.csv')

# Assuming the labels are the same for both PCA and PLS data
labels = df_pca.iloc[:, -1].values

# Extract PCA scores and PLS factors
PC1 = df_pca['PC1'].values
PC2 = df_pca['PC2'].values
Factor1 = df_pls['Factor1'].values
Factor2 = df_pls['Factor2'].values

# Convert labels to numeric format using LabelEncoder
label_encoder = LabelEncoder()
y_numeric = label_encoder.fit_transform(labels)

# Create a mapping from numeric labels to sample names
label_mapping = dict(zip(y_numeric, labels))

# Combine PCA and PLS components for t-SNE
combined_data = np.column_stack((PC1, PC2, Factor1, Factor2))

# Apply T-SNE on the combined data
tsne = TSNE(n_components=2, perplexity=5, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(combined_data)

# Plotting the T-SNE result
plt.figure(figsize=(14, 10))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_numeric, cmap='viridis', alpha=0.6)

# Add labels to each point
for i, txt in enumerate(labels):
    plt.text(X_tsne[i, 0], X_tsne[i, 1], txt, fontsize=6, ha='right')

# Create a colorbar with both encoded labels and sample names
cbar = plt.colorbar(scatter, ticks=np.arange(len(label_mapping)))
cbar.set_label('Encoded Labels')
cbar.set_ticks(list(label_mapping.keys()))
cbar.set_ticklabels([f"{key} - {value}" for key, value in label_mapping.items()])

# Adjust the font size of the colorbar labels
cbar.ax.tick_params(labelsize=8)

plt.title('T-SNE Visualization of Combined PCA and PLS Components with Sample Names')
plt.xlabel('T-SNE Component 1')
plt.ylabel('T-SNE Component 2')
plt.grid(True)
plt.show()
