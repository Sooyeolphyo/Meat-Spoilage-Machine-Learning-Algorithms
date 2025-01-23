Install Required LibrariesRun the following command to install the necessary Python libraries:

pip install pandas numpy scikit-learn matplotlib

Prepare DataEnsure that your_pca_factors2.csv (PCA data) and your_pls_factors2.csv (PLS data) are available in the specified paths.Example structure:

- **PCA data (`your_pca_factors2.csv`)**:

PC1, PC2
0.12, 0.34
0.56, 0.78

- **PLS data (`your_pls_factors2.csv`)**:

Factor2, Factor3
1.23, 4.56
7.89, 2.34

Run the ScriptExecute the Python script:

python K-mean, KNN.py

Output

Elbow Graph: Helps determine the optimal number of clusters.

K-Means Results: Visualizes the clustered data.

K-NN Accuracy: Calculates the classification accuracy on the test data.

Visualizations: 2D plots for clustering and classification results.


