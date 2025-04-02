import os
os.environ['OMP_NUM_THREADS'] = '1'

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.cluster import KMeans

# Numerical and categorical features in spotify dataset
n_features = ['artist_count', 'in_spotify_playlists', 'streams', 'in_apple_playlists', 'in_deezer_playlists', 'bpm', 'danceability', 'valence', 'energy', 'acousticness', 'instrumentalness', 'liveness', 'speechiness']
c_features = ['key', 'mode']
# Load the dataset
df = pd.read_csv('spotify-2023.csv', usecols=n_features + c_features)

# Drop rows that can't be parsed as numbers
df['streams'] = pd.to_numeric(df['streams'], errors='coerce')
df = df.dropna(subset=['streams'])
df['streams'] = df['streams'].astype(int)

df['in_deezer_playlists'] = pd.to_numeric(df['in_deezer_playlists'], errors='coerce')
df = df.dropna(subset=['in_deezer_playlists'])
df['in_deezer_playlists'] = df['in_deezer_playlists'].astype(int)

transformer = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), n_features),
        ('cat', OneHotEncoder(), c_features)
    ],
    remainder='passthrough'
)

# Standardize the features
X = transformer.fit_transform(df)

# Run KMeans clustering
cluster_ids = pd.DataFrame()

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)

    labels = kmeans.labels_

    cluster_ids['kmeans_' + str(i)] = labels

df['mode'] = df['mode'].astype('category').cat.codes
df['key'] = df['key'].astype('category').cat.codes

result = pd.concat([df, cluster_ids], axis=1)
result.to_csv('./static/full-spotify-clustered.csv', index=False)
