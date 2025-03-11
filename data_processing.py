import pandas as pd
import csv

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error

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
pca = PCA()
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

# Run KMeans clustering
import os

os.environ['OMP_NUM_THREADS'] = '1'

mse_scores = []
cluster_ids = pd.DataFrame()

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    cluster_ids['kmeans_' + str(i)] = labels

    mse = mean_squared_error(X, centroids[labels])
    mse_scores.append(mse)

# print("MSE scores:", mse_scores)
# print("Cluster IDs:\n", cluster_ids)
