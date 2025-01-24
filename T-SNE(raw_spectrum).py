import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load your data
df = pd.read_csv('c:/Startcoding/your_data.csv')

# Assuming the last column contains the labels (days of decomposition)
num_features = 14934
X = df.iloc[:, :num_features].values
y = df.iloc[:, num_features].values

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert labels to numeric format using LabelEncoder
label_encoder = LabelEncoder()
y_numeric = label_encoder.fit_transform(y)

# Apply T-SNE
tsne = TSNE(n_components=2, perplexity=5, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Plotting the T-SNE result
plt.figure(figsize=(14, 10))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_numeric, cmap='viridis', alpha=0.6)

# Add labels to each point
for i, txt in enumerate(y):
    plt.text(X_tsne[i, 0], X_tsne[i, 1], txt, fontsize=9, ha='right')

plt.colorbar(scatter, label='Encoded Labels')
plt.title('T-SNE Visualization of IR Spectrum Data with Sample Names')
plt.xlabel('T-SNE Component 1')
plt.ylabel('T-SNE Component 2')
plt.grid(True)
plt.show()
