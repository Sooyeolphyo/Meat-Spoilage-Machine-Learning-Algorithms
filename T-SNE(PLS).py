import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

# Load your PLS factor data
df_factors = pd.read_csv('c:/Startcoding/your_pls_factors.csv')

# Assuming Factor 1 and Factor 2 are in the second and third columns (index 1 and 2)
X_pls_selected = df_factors.iloc[:, [1, 2]].values  # Select only Factor 1 and Factor 2
y = df_factors.iloc[:, -1].values  # Assuming the last column contains the labels

# Convert labels to numeric format using LabelEncoder
label_encoder = LabelEncoder()
y_numeric = label_encoder.fit_transform(y)

# Create a mapping from numeric labels to sample names
label_mapping = dict(zip(y_numeric, y))

# Apply T-SNE on the selected PLS factors (Factor 1 and Factor 2)
tsne = TSNE(n_components=2, perplexity=5, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_pls_selected)

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

plt.title('T-SNE Visualization of Selected PLS Factors (Factor 1 and Factor 2) with Sample Names')
plt.xlabel('T-SNE Component 1')
plt.ylabel('T-SNE Component 2')
plt.grid(True)
plt.show()
