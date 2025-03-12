import numpy as np
import higra as hg
import pandas as pd
from fcmeans import FCM
from astropy.io import fits
from astropy.wcs import WCS
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler


def read_image_data(file_path):

    with fits.open(file_path) as hdu_list:

        image_hdu = next((hdu for hdu in hdu_list if hdu.data is not None), None)

        if image_hdu is None:
            raise ValueError("No valid image data found in the FITS file.")

        image = image_hdu.data
        header = image_hdu.header.copy()

    return image, header


def image_value_check(image):

    if np.isnan(image).any():
        min_value = np.nanmin(image)
        image[np.isnan(image)] = min_value

    return np.abs(image)


def image_value_check_2(image):

    if np.isnan(image).any():
        finite_values = image[np.isfinite(image)]
        if finite_values.size == 0:
            raise ValueError("Image contains no finite values (only NaNs or infinities).")
        min_value = finite_values.min()
        image = np.where(np.isnan(image), min_value, image)

    if np.isinf(image).any():
        finite_values = image[np.isfinite(image)]
        if finite_values.size == 0:
            raise ValueError("Image contains no finite values (only infinities or NaNs).")
        max_value = finite_values.max()
        min_value = finite_values.min()
        image = np.where(image == np.inf, max_value, image)
        image = np.where(image == -np.inf, min_value, image)

    return image


def smooth_filter(image, sigma=2):

    return gaussian_filter(image, sigma)


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

    #TODO: check "I_0 = mean - intensity"
    I_0 = mean
    np.nan_to_num(variance, nan=0)
    sigma = np.sqrt(np.maximum(variance, 0))  # Ensure non-negative variance
    gaussian_intensity = gaussian_profile(I_0, sigma, distances, mu=center)

    return gaussian_intensity


def save_fits_with_header(data, header, output_path):

    hdu = fits.PrimaryHDU(data, header=header)
    hdu.writeto(output_path, overwrite=True)

    return None


def sky_coordinates(y, x, header):

    wcs = WCS(header)
    pixel_coords = np.array([x, y]).T
    sky_coords = wcs.all_pix2world(pixel_coords, 1)
    ra = sky_coords[:, 0]
    dec = sky_coords[:, 1]

    return ra, dec


def save_parameters(id, x, y, ra, dec, flux, flux_calibrated, area, a, b, theta, file_name):

    parameters = {
        "Segment_ID": id,
        "X":x,
        "Y": y,
        "RA": ra,
        "DEC": dec,
        "Flux": flux,
        "Flux_Calibrated": flux_calibrated,
        "Area": area,
        "a": a,
        "b": b,
        "Theta": theta,
    }
    parameters_df = pd.DataFrame(parameters)
    parameters_df.to_csv(file_name, index=False)

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


def attribute_statistical_significance(tree, altitudes, volume, area, background_var, gain, alpha=1e-6):

    denominator = background_var + altitudes[tree.parents()] / gain
    safe_denominator = np.where(denominator == 0, np.finfo(float).eps, denominator)
    volume /= safe_denominator

    significant_nodes = volume > chi2.ppf(alpha, area)
    significant_nodes[:tree.num_leaves()] = False

    return significant_nodes


def attribute_main_branch(tree):

    area = hg.attribute_area(tree)
    largest_child = hg.accumulate_parallel(tree, area, hg.Accumulators.argmax)
    child_number = hg.attribute_child_number(tree)

    return child_number == largest_child[tree.parents()]


def select_objects(tree, significant_nodes):

    filtered_tree, node_map = hg.simplify_tree(tree, np.logical_not(significant_nodes))
    main_branch = attribute_main_branch(filtered_tree)

    if not significant_nodes[tree.root()]:
        root_children = filtered_tree.children(filtered_tree.root())
        main_branch[root_children] = False

    res = np.zeros(tree.num_vertices(), dtype=np.bool_)
    res[node_map] = np.logical_not(main_branch)

    return res


def move_up(
    tree, altitudes, area, parent_area, distances, objects, background_var, gain, gamma_distance, gaussian, move_factor
):

    main_branch = attribute_main_branch(tree)

    closest_object_ancestor = hg.propagate_sequential(
        tree,
        np.arange(tree.num_vertices()),
        np.logical_and(main_branch, np.logical_not(objects)))

    target_altitudes = altitudes.copy()
    object_indexes, = np.nonzero(objects)
    local_noise = np.sqrt(background_var + altitudes[tree.parent(object_indexes)] / gain)
    target_altitudes[object_indexes] = altitudes[object_indexes] + move_factor * local_noise

    target_altitudes = target_altitudes[closest_object_ancestor]
    valid_moves = np.logical_and(
        np.logical_and(
            altitudes >= target_altitudes,
            np.logical_and(
                np.sqrt(area/np.pi) > distances,
                np.logical_and(
                    objects[closest_object_ancestor],
                    area/parent_area >= .78
                )
            )
        ),
        #TODO: check: "altitudes/area>=gaussian"
        altitudes>=gaussian,
    )

    parent_closest_object_ancestor = closest_object_ancestor[tree.parents()]
    parent_not_valid_moves = np.logical_not(valid_moves[tree.parents()])
    new_objects = np.logical_and(
        valid_moves,
        np.logical_or(
            parent_not_valid_moves,
            parent_closest_object_ancestor != closest_object_ancestor
        )
    )

    return new_objects

