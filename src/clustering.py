# clustering.py

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def perform_clustering(X):

    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X)

    score = silhouette_score(X, clusters)

    return kmeans, clusters, score