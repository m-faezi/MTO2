import numpy as np
from fcmeans import FCM
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler



def fuzz_bg_structure(bg_candidate_features, main_branch, altitudes):

    masked_features = np.vstack(bg_candidate_features).T
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(masked_features)
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(scaled_features)
    fcm = FCM(n_clusters=2)
    fcm.fit(reduced_features)
    labels = fcm.predict(reduced_features)
    labels_array = np.array(labels)
    all_labels = np.zeros(altitudes.size)
    all_labels[~main_branch] = labels_array

    return all_labels


def binary_cluster_bg_structure(bg_candidate_features, main_branch, altitudes):

    masked_features = np.vstack(bg_candidate_features).T
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(masked_features)
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(scaled_features)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(reduced_features)
    labels = kmeans.labels_
    all_labels = np.zeros(altitudes.size, dtype=int)
    all_labels[~main_branch] = labels

    return all_labels


def binary_cluster_bg_structure_minibatch(bg_candidate_features, main_branch, altitudes):

    masked_features = np.vstack(bg_candidate_features).T
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(masked_features)
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(scaled_features)
    kmeans = MiniBatchKMeans(n_clusters=2, random_state=42, batch_size=256, max_iter=100)
    kmeans.fit(reduced_features)
    labels = kmeans.labels_
    all_labels = np.zeros(altitudes.size, dtype=int)
    all_labels[~main_branch] = labels

    return all_labels


