import numpy as np

# Sample data (replace with your dataset)
data = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])

# Step 1: Mean centering
mean = np.mean(data, axis=0)
centered_data = data - mean

# Step 2: Compute the covariance matrix
cov_matrix = np.cov(centered_data, rowvar=False)

# Step 3: Eigendecomposition of the covariance matrix
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Sort eigenvectors by decreasing eigenvalues
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Step 4: Select a subset of eigenvectors (principal components)
num_components = 2  # Choose the number of components to keep
selected_eigenvectors = eigenvectors[:, :num_components]

# Step 5: Project the data onto the selected principal components
reduced_data = np.dot(centered_data, selected_eigenvectors)

# Now, 'reduced_data' contains the data in reduced dimensions

# You can also calculate the variance explained by the selected components
total_variance = np.sum(eigenvalues)
explained_variance = np.sum(eigenvalues[:num_components])
explained_variance_ratio = explained_variance / total_variance

print(f"Explained Variance Ratio: {explained_variance_ratio:.4f}")
