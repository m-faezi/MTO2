import torch
from sklearn.preprocessing import StandardScaler
import numpy as np


def pytorch_kmeans_bg_structure(
        bg_candidate_features,
        main_branch,
        altitudes,
        max_iters=100,
        tol=0
):

    masked_features = np.vstack(bg_candidate_features).T
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(masked_features)

    features = torch.FloatTensor(scaled_features)

    k = 2
    n_samples, n_features = features.shape
    indices = torch.randperm(n_samples)[:k]
    centroids = features[indices]

    for iteration in range(max_iters):

        distances = torch.cdist(features, centroids)
        labels = torch.argmin(distances, dim=1)

        new_centroids = torch.zeros_like(centroids)

        for i in range(k):

            mask = labels == i

            if mask.sum() > 0:

                new_centroids[i] = features[mask].mean(dim=0)

        centroid_shift = torch.norm(new_centroids - centroids)

        if centroid_shift < tol:

            break

        centroids = new_centroids

    labels_np = labels.numpy()
    all_labels = np.zeros(altitudes.size, dtype=int)
    all_labels[~main_branch] = labels_np

    return all_labels


def pytorch_fuzzy_c_means(
        bg_candidate_features,
        main_branch,
        altitudes,
        m=2.0,
        max_iters=100,
        tol=0
):

    masked_features = np.vstack(bg_candidate_features).T
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(masked_features)

    features = torch.FloatTensor(scaled_features)
    n_samples, n_features = features.shape
    k = 2

    U = torch.rand(n_samples, k)
    U = U / U.sum(dim=1, keepdim=True)

    for iteration in range(max_iters):

        U_m = U ** m
        centroids = (U_m.t() @ features) / U_m.sum(dim=0, keepdim=True).t()

        distances = torch.cdist(features, centroids)

        power = 2.0 / (m - 1)

        with torch.no_grad():

            new_U = 1.0 / (distances ** power)
            new_U = new_U / new_U.sum(dim=1, keepdim=True)

        U_shift = torch.norm(new_U - U)

        if U_shift < tol:

            break

        U = new_U

    labels = torch.argmax(U, dim=1)
    labels_np = labels.numpy()

    all_labels = np.zeros(altitudes.size, dtype=int)
    all_labels[~main_branch] = labels_np

    return all_labels

