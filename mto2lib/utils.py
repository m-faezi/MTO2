
import numpy as np
import higra as hg
import pandas as pd
from fcmeans import FCM
from astropy.io import fits
from astropy.wcs import WCS
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler














def image_to_hierarchical_structure(image):
    
    graph_structure = hg.get_8_adjacency_graph(image.shape)
    tree_structure, altitudes = hg.component_tree_max_tree(graph_structure, image)

    return graph_structure, tree_structure, altitudes


def centroid(tree, size):

    emb = hg.EmbeddingGrid2d(size)
    coord = emb.lin2grid(np.arange(tree.num_leaves()))
    m = np.zeros((tree.num_leaves(), 3), dtype=np.float64)
    m[:, 0] = 1
    m[:, 1] = coord[:, 0]
    m[:, 2] = coord[:, 1]
    m = hg.accumulate_sequential(tree, m, hg.Accumulators.sum)
    m00 = m[:, 0]
    m10 = m[:, 1]
    m01 = m[:, 2]
    x_mean = np.divide(m10, m00, out=np.zeros_like(m10), where=m00 != 0)
    y_mean = np.divide(m01, m00, out=np.zeros_like(m01), where=m00 != 0)

    return x_mean, y_mean


def weighted_centroid(tree, size, image):

    image -= max(np.min(image), 0)
    im = np.exp(image)
    emb = hg.EmbeddingGrid2d(size)
    coord = emb.lin2grid(np.arange(tree.num_leaves()))

    m = np.zeros((tree.num_leaves(), 3), dtype=np.float64)
    m[:, 0] = 1
    m[:, 1] = coord[:, 0]
    m[:, 2] = coord[:, 1]
    m = np.array(
        [[im[int(i[1]), int(i[2])], i[1] * im[int(i[1]), int(i[2])], i[2] * im[int(i[1]), int(i[2])]] for i in m]
    )
    m = hg.accumulate_sequential(tree, m, hg.Accumulators.sum)
    m00 = m[:, 0]
    m10 = m[:, 1]
    m01 = m[:, 2]

    x_mean = np.divide(m10, m00, out=np.zeros_like(m10), where=m00 != 0)
    y_mean = np.divide(m01, m00, out=np.zeros_like(m01), where=m00 != 0)

    return x_mean, y_mean


def weighted_centroid_2(tree, size, image):

    """Calculate flux-weighted centroids with proper normalization"""
    # Ensure positive weights
    weights = np.maximum(image - np.min(image), 0) + 1e-10  # Small epsilon to avoid division by zero
    emb = hg.EmbeddingGrid2d(size)
    coord = emb.lin2grid(np.arange(tree.num_leaves()))

    # Calculate weighted sums
    m = np.zeros((tree.num_leaves(), 3))
    m[:, 0] = weights.ravel()  # Flux weights
    m[:, 1] = coord[:, 0] * weights.ravel()  # X-weighted
    m[:, 2] = coord[:, 1] * weights.ravel()  # Y-weighted

    # Accumulate through tree
    m = hg.accumulate_sequential(tree, m, hg.Accumulators.sum)

    # Calculate centroids
    x_mean = np.divide(m[:, 1], m[:, 0], out=np.zeros_like(m[:, 1]), where=m[:, 0] != 0)
    y_mean = np.divide(m[:, 2], m[:, 0], out=np.zeros_like(m[:, 2]), where=m[:, 0] != 0)

    return x_mean, y_mean


def weighted_centroid_coords_from_segments(image, coords):
    """
    Compute the flux-weighted centroid for a set of coordinates.

    Args:
        image: 2D numpy array of pixel values
        coords: List of (y, x) coordinate tuples

    Returns:
        (y_centroid, x_centroid): Flux-weighted center coordinates
    """
    if not coords:
        return (0.0, 0.0)

    coords_array = np.array(coords)
    y = coords_array[:, 0]
    x = coords_array[:, 1]


    intensities = image[y, x]
    total_flux = np.sum(intensities)

    if total_flux == 0:

        return (np.mean(y), np.mean(x))

    y_centroid = np.sum(y * intensities) / total_flux
    x_centroid = np.sum(x * intensities) / total_flux

    return (y_centroid, x_centroid)


def get_max_tree_attributes(tree_structure, altitudes, image):

    area = hg.attribute_area(tree_structure)
    volume = hg.attribute_volume(tree_structure, altitudes, area)
    mean, variance = hg.attribute_gaussian_region_weights_model(tree_structure, image)
    topological_height = hg.attribute_topological_height(tree_structure)
    x, y = centroid(tree_structure, image.shape[:2])
    distance_to_root_center = np.sqrt((x[tree_structure.root()] - x) ** 2 + (y[tree_structure.root()] - y) ** 2)

    return area, volume, mean, variance, topological_height, distance_to_root_center


def fuzz_bg_structure(bg_candidate_features, non_bool_unique_topological_height, altitudes):

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
    all_labels[~non_bool_unique_topological_height] = labels_array

    return all_labels


def half_light_radius(image, coords):

    if len(coords) == 0:
        return 0

    intensities = np.array([image[y, x] for y, x in coords])
    total_flux_ = np.sum(intensities)

    if total_flux_ == 0:
        return 0

    y0, x0 = np.mean(coords, axis=0)
    radii = np.sqrt((np.array(coords)[:, 0] - y0)**2 + (np.array(coords)[:, 1] - x0)**2)

    sorted_indices = np.argsort(radii)
    sorted_radii = radii[sorted_indices]
    sorted_flux = intensities[sorted_indices]

    cumulative_flux = np.cumsum(sorted_flux)

    hlr_index = np.searchsorted(cumulative_flux, total_flux_ / 2.0)
    return sorted_radii[hlr_index] if hlr_index < len(sorted_radii) else sorted_radii[-1]


def compute_r_fwhm(image, coords):
    """Compute FWHM radius (fast, no subsampling)."""
    if not coords:
        return 0.0

    # Vectorized intensity lookup and peak detection
    y, x = np.array(coords).T
    intensities = image[y, x]
    peak = np.max(intensities)
    if peak == 0:
        return 0.0

    # Flux-weighted centroid (1 pass)
    flux = np.sum(intensities)
    yc = np.sum(y * intensities) / flux
    xc = np.sum(x * intensities) / flux

    # Mask pixels >= half-max and compute mean distance
    mask = intensities >= peak * 0.5
    if not np.any(mask):
        return 0.0

    r_fwhm = np.mean(np.sqrt((y[mask] - yc)**2 + (x[mask] - xc)**2))

    return r_fwhm


def binary_cluster_bg_structure(bg_candidate_features, non_bool_unique_topological_height, altitudes):

    masked_features = np.vstack(bg_candidate_features).T
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(masked_features)
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(scaled_features)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(reduced_features)
    labels = kmeans.labels_
    all_labels = np.zeros(altitudes.size, dtype=int)
    all_labels[~non_bool_unique_topological_height] = labels

    return all_labels


def binary_cluster_bg_structure_minibatch(bg_candidate_features, non_bool_unique_topological_height, altitudes):

    masked_features = np.vstack(bg_candidate_features).T
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(masked_features)
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(scaled_features)
    kmeans = MiniBatchKMeans(n_clusters=2, random_state=42, batch_size=256, max_iter=100)
    kmeans.fit(reduced_features)
    labels = kmeans.labels_
    all_labels = np.zeros(altitudes.size, dtype=int)
    all_labels[~non_bool_unique_topological_height] = labels

    return all_labels


def mark_non_unique(array):

    unique, counts = np.unique(array, return_counts=True)
    count_dict = dict(zip(unique, counts))
    counts_array = np.vectorize(count_dict.get)(array)
    bool_array = counts_array > 1

    return bool_array


def gaussian_profile(I_0, sigma, R, mu=0):

    epsilon = np.finfo(np.float64).eps
    sigma = np.maximum(sigma, epsilon)

    return I_0 * np.exp(-((R - mu) ** 2) / (2 * sigma ** 2))


def compute_gaussian_profile(mean, variance, distances, intensity, center=0):

    I_0 = intensity
    np.nan_to_num(variance, nan=0)
    sigma = np.sqrt(np.maximum(variance, 0))
    gaussian_intensity = gaussian_profile(I_0, sigma, distances, mu=center)

    return gaussian_intensity





def sky_coordinates(y, x, header):

    wcs = WCS(header)
    pixel_coords = np.array([x, y]).T
    sky_coords = wcs.all_pix2world(pixel_coords, 1)
    ra = sky_coords[:, 0]
    dec = sky_coords[:, 1]

    return ra, dec


def save_parameters(id, x, y, ra, dec, flux, flux_calibrated, area, a, b, theta, r_eff, r_fwhm,  file_name):

    parameters_df = pd.DataFrame(
        {
            "Segment_ID": id[1:],
            "X": x[1:],
            "Y": y[1:],
            "RA": ra[1:],
            "DEC": dec[1:],
            "Flux": flux[1:],
            "Flux_Calibrated": flux_calibrated[1:],
            "Area": area[1:],
            "a": a[1:],
            "b": b[1:],
            "Theta": theta[1:],
            "R_eff": r_eff[1:],
            "R_fwhm": r_fwhm[1:],
        }
    )

    parameters_df.to_csv(str(file_name)+".csv", index=False)

    return None


def total_flux(tree, size, image):

    emb = hg.EmbeddingGrid2d(size)
    coord = emb.lin2grid(np.arange(tree.num_leaves()))

    m = np.zeros((tree.num_leaves(), 3), dtype=np.float64)
    m[:, 0] = 1
    m[:, 1] = coord[:, 0]
    m[:, 2] = coord[:, 1]
    m = np.array([[
        image[int(i[1]), int(i[2])],
        i[1] * image[int(i[1]), int(i[2])],
        i[2] * image[int(i[1]), int(i[2])]
    ] for i in m])
    m = hg.accumulate_sequential(tree, m, hg.Accumulators.sum)
    t_f = m[:, 0]

    return t_f


def second_order_moments(tree, size, image):

    image -= max(np.min(image), 0)
    im = np.exp(image)
    emb = hg.EmbeddingGrid2d(size)
    coord = emb.lin2grid(np.arange(tree.num_leaves()))
    m = np.zeros((tree.num_leaves(), 6), dtype=np.float64)
    m[:, 0] = 1
    m[:, 1] = coord[:, 0]
    m[:, 2] = coord[:, 1]
    m[:, 3] = coord[:, 0] ** 2
    m[:, 4] = coord[:, 1] ** 2
    m[:, 5] = coord[:, 0] * coord[:, 1]

    m = np.array([[im[int(i[1]), int(i[2])],
                   i[1] * im[int(i[1]), int(i[2])],
                   i[2] * im[int(i[1]), int(i[2])],
                   (i[1] ** 2) * im[int(i[1]), int(i[2])],
                   (i[2] ** 2) * im[int(i[1]), int(i[2])],
                   (i[1] * i[2]) * im[int(i[1]), int(i[2])]] for i in m])

    m = hg.accumulate_sequential(tree, m, hg.Accumulators.sum)
    m00 = m[:, 0]
    m10 = m[:, 1]
    m01 = m[:, 2]
    x_mean = m10 / m00
    y_mean = m01 / m00
    x2 = (m[:, 3] / m00) - (x_mean ** 2)
    y2 = (m[:, 4] / m00) - (y_mean ** 2)
    xy = (m[:, 5] / m00) - (x_mean * y_mean)
    lhs = (x2 + y2) / 2
    rhs = np.sqrt(np.power((x2 - y2) / 2, 2) + np.power(xy, 2))
    major_axis = np.sqrt(lhs + rhs)
    minor_axis = np.sqrt(np.maximum(lhs - rhs, 0))
    t = np.arctan2(2 * xy, (x2 - y2))
    theta = np.where(
        (xy < 0) & (t > 0),
        (t - np.pi) / 2,
        np.where(
            (t < 0) & (xy > 0),
            (t + np.pi) / 2, t / 2
        )
    )

    return major_axis, minor_axis, theta

