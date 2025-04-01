import os
os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd
import csv

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

from kneed import KneeLocator

from sklearn.manifold import MDS

# Numerical features in spotify dataset
features = ['artist_count', 'in_spotify_playlists', 'streams', 'in_apple_playlists', 'in_deezer_playlists', 'bpm', 'danceability', 'valence', 'energy', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']

# Load the dataset
df = pd.read_csv('spotify-2023.csv', usecols=features)

# Drop rows that can't be parsed as numbers
df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
df = df.dropna(subset=['streams'])
df['streams'] = df['streams'].astype(int)

df['in_deezer_playlists'] = pd.to_numeric(df['in_deezer_playlists'], errors='coerce')
df = df.dropna(subset=['in_deezer_playlists'])
df['in_deezer_playlists'] = df['in_deezer_playlists'].astype(int)

# Standardize the features
X = StandardScaler().fit_transform(df)

# Apply PCA
pca = PCA(random_state=42)
pca_values = pca.fit_transform(X)

eigenvalues = pca.explained_variance_
eigenvectors = pca.components_

# print("Eigenvalues:", eigenvalues)
# print("Eigenvectors:\n", eigenvectors)

# Write the eigenvalues to a CSV file
with open('./static/scree.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['x', 'y'])
    for x, y in enumerate(eigenvalues, start=1):
        writer.writerow([x, y])

# Find the elbow of the scree plot
kneedle = KneeLocator(range(1, len(eigenvalues)+1), eigenvalues, S=1.0, curve='convex', direction='decreasing')
# print(round(kneedle.elbow))

# Run KMeans clustering
mse_scores = []
cluster_ids = pd.DataFrame()

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    cluster_ids['kmeans_' + str(i)] = labels

    mse = mean_squared_error(X, centroids[labels])
    mse_scores.append(mse)

# print("MSE scores:", mse_scores)
# print("Cluster IDs:\n", cluster_ids)

# Write the MSE scores to a CSV file
with open('./static/mse_scores.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['k', 'mse'])
    for k, mse in enumerate(mse_scores, start=1):
        writer.writerow([k, mse])

# Find the elbow of the MSE plot
kneedle = KneeLocator(range(1, 11), mse_scores, S=1.0, curve='convex', direction='decreasing')
# print(round(kneedle.elbow))

# Write the PCA values and kmeans color to a CSV file
with open('./static/biplot.csv', 'w', newline='') as f:
    labels = []
    for i in range(len(eigenvectors)):
        labels.append('PC' + str(i+1))
    for i in range(10):
        labels.append('kmeans_' + str(i+1))
    writer = csv.writer(f)
    writer.writerow(labels)
    for i in range(len(pca_values)):
        row = []
        for j in range(len(eigenvectors)):
            row.append(pca_values[i][j])
        for j in range(10):
            row.append(cluster_ids['kmeans_' + str(j+1)][i])
        writer.writerow(row)


# Write the eigenvectors and labels to a CSV file
with open('./static/loadings.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['feature', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13'])
    for i, feature in enumerate(features):
        row = [feature]
        for j in range(len(eigenvectors)):
            row.append(eigenvectors[j][i])
        writer.writerow(row)

result = pd.concat([df, cluster_ids], axis=1)
result.to_csv('./static/spotify-clustered.csv', index=False)

# Run MDS on dataset and write the output and cluster ids to a CSV file
embedding = MDS(n_components=2, dissimilarity='euclidean', random_state=42)
mds = embedding.fit_transform(df)
with open('./static/data_mds.csv', 'w', newline='') as f:
    labels = ['x', 'y']
    for i in range(10):
        labels.append('kmeans_' + str(i+1))
    writer = csv.writer(f)
    writer.writerow(labels)
    for i in range(len(mds)):
        row = [mds[i][0], mds[i][1]]
        for j in range(10):
            row.append(cluster_ids['kmeans_' + str(j+1)][i])
        writer.writerow(row)


embedding = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
mds = embedding.fit_transform(1-abs(df.corr()))
with open('./static/var_mds.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['x', 'y', 'feature'])
    for i in range(len(mds)):
        row = [mds[i][0], mds[i][1], features[i]]
        writer.writerow(row)
