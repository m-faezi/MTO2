import os
import numpy
import numpy as np
import higra as hg
from fcmeans import FCM
from astropy.io import fits
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter
from sklearn.preprocessing import StandardScaler
from astropy.wcs import WCS
import pandas as pd
from sklearn.cluster import KMeans


def read_image_data(file_path, y_min=0, y_max=-1, x_min=0, x_max=-1):
    with fits.open(file_path) as hdu_list:
        # Attempt to get the first valid image HDU
        image_hdu = None
        for hdu in hdu_list:
            if hdu.data is not None:
                image_hdu = hdu
                break

        if image_hdu is None:
            raise ValueError("No valid image data found in the FITS file.")

        # Handle negative slicing values
        full_image = image_hdu.data
        if y_max == -1:
            y_max = full_image.shape[0]  # Use full height
        if x_max == -1:
            x_max = full_image.shape[1]  # Use full width

        # Ensure indices are within bounds
        y_min = max(0, y_min)
        y_max = min(full_image.shape[0], y_max)
        x_min = max(0, x_min)
        x_max = min(full_image.shape[1], x_max)

        # Slice the image data
        image = full_image[y_min:y_max, x_min:x_max]

        # Update header information
        header = image_hdu.header.copy()

        # Adjust CRPIX1 and CRPIX2 (reference pixel) for the new crop
        if 'CRPIX1' in header and 'CRPIX2' in header:
            header['CRPIX1'] -= x_min
            header['CRPIX2'] -= y_min

        # Update the NAXIS1 and NAXIS2 to reflect the new image size
        header['NAXIS1'] = image.shape[1]
        header['NAXIS2'] = image.shape[0]

    # Return the cropped image and the updated header
    return image, header


def image_value_check(image):

    # image[image < 0] = 0

    if np.isnan(image).any():
        min_value = np.nanmin(image)
        image[np.isnan(image)] = min_value

    return image


def image_value_check_2(image):

    # Check for NaN values.
    if np.isnan(image).any():
        finite_values = image[np.isfinite(image)]
        if finite_values.size == 0:
            raise ValueError("Image contains no finite values (only NaNs or infinities).")
        min_value = finite_values.min()
        image = np.where(np.isnan(image), min_value, image)

    # Check for infinite values.
    if np.isinf(image).any():
        finite_values = image[np.isfinite(image)]
        if finite_values.size == 0:
            raise ValueError("Image contains no finite values (only infinities or NaNs).")
        max_value = finite_values.max()
        min_value = finite_values.min()
        # Replace positive infinity with the maximum finite value.
        image = np.where(image == np.inf, max_value, image)
        # Replace negative infinity with the minimum finite value.
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
    x_mean = m10 / m00
    y_mean = m01 / m00

    return x_mean, y_mean


def weighted_centroid(tree, size, image):

    image -= max(np.min(image), 0)
    im = numpy.exp(image)
    emb = hg.EmbeddingGrid2d(size)
    coord = emb.lin2grid(np.arange(tree.num_leaves()))

    m = np.zeros((tree.num_leaves(), 3), dtype=np.float64)
    m[:, 0] = 1
    m[:, 1] = coord[:, 0]
    m[:, 2] = coord[:, 1]
    m = numpy.array(
        [[im[int(i[1]), int(i[2])], i[1] * im[int(i[1]), int(i[2])], i[2] * im[int(i[1]), int(i[2])]] for i in m]
    )
    m = hg.accumulate_sequential(tree, m, hg.Accumulators.sum)
    m00 = m[:, 0]
    m10 = m[:, 1]
    m01 = m[:, 2]

    x_mean = m10 / m00
    y_mean = m01 / m00

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

    # Apply PCA to reduce dimensionality if necessary (optional)
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(scaled_features)

    # Fuzzy clustering
    fcm = FCM(n_clusters=2)
    fcm.fit(reduced_features)

    # Get the labels of the binary clusters
    labels = fcm.predict(reduced_features)

    # Return the labels
    labels_array = np.array(labels)

    all_labels = np.zeros(altitudes.size)
    all_labels[~non_bool_unique_topological_height] = labels_array

    return all_labels


def binary_cluster_bg_structure(bg_candidate_features, non_bool_unique_topological_height, altitudes):
    # Stack features and transpose for clustering
    masked_features = np.vstack(bg_candidate_features).T

    # Standardize the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(masked_features)

    # Optional: Reduce dimensionality using PCA
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(scaled_features)

    # Apply K-Means clustering with 2 clusters
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans.fit(reduced_features)

    # Get cluster labels
    labels = kmeans.labels_

    # Map labels to the original dataset
    all_labels = np.zeros(altitudes.size, dtype=int)
    all_labels[~non_bool_unique_topological_height] = labels

    return all_labels


from sklearn.cluster import MiniBatchKMeans


def binary_cluster_bg_structure_minibatch(bg_candidate_features, non_bool_unique_topological_height, altitudes):
    masked_features = np.vstack(bg_candidate_features).T
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(masked_features)

    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(scaled_features)

    # Use MiniBatchKMeans instead of KMeans
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
    """Calculate the Gaussian profile with robust handling."""
    epsilon = 1e-10  # Small value to avoid numerical instability
    sigma = np.maximum(sigma, epsilon)  # Avoid division by zero
    return I_0 * np.exp(-((R - mu) ** 2) / (2 * sigma ** 2))

def compute_gaussian_profile(mean, variance, distances, center=0):
    """Compute the Gaussian model for intensity."""
    I_0 = mean
    sigma = np.sqrt(np.maximum(variance, 0))  # Ensure non-negative variance
    gaussian_intensity = gaussian_profile(I_0, sigma, distances, mu=center)
    return gaussian_intensity

def compute_gaussian_profile_2(mean, variance, distances, intensity, center=0):
    """Compute the Gaussian model for intensity."""
    I_0 = mean - intensity
    sigma = np.sqrt(np.maximum(variance, 0))  # Ensure non-negative variance
    gaussian_intensity = gaussian_profile(I_0, sigma, distances, mu=center)
    return gaussian_intensity


def save_fits_with_header(data, header, output_path):
    # Create a PrimaryHDU object using the data and header
    hdu = fits.PrimaryHDU(data, header=header)
    # Write to the specified FITS file
    hdu.writeto(output_path, overwrite=True)


def sky_coordinates(y, x, header):
    # Extract RA and Dec
    wcs = WCS(header)
    pixel_coords = np.array([x, y]).T
    sky_coords = wcs.all_pix2world(pixel_coords, 1)

    ra = sky_coords[:, 0]
    dec = sky_coords[:, 1]

    return ra, dec


def save_parameters(id, x, y, ra, dec, flux, flux_calibrated, area, a, b, theta):
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
    parameters_df.to_csv('output/parameters.csv', index=False)


def total_flux(tree, size, image):
    emb = hg.EmbeddingGrid2d(size)
    coord = emb.lin2grid(np.arange(tree.num_leaves()))

    m = np.zeros((tree.num_leaves(), 3), dtype=np.float64)
    m[:, 0] = 1
    m[:, 1] = coord[:, 0]
    m[:, 2] = coord[:, 1]
    m = numpy.array([[
        image[int(i[1]), int(i[2])],
        i[1] * image[int(i[1]), int(i[2])],
        i[2] * image[int(i[1]), int(i[2])]
    ] for i in m])
    m = hg.accumulate_sequential(tree, m, hg.Accumulators.sum)
    t_f = m[:, 0]

    return t_f


def second_order_moments(tree, size, image):
    """
    Calculate the second-order moments for each node in the tree, returning arrays of
    major axes, minor axes, and orientation angles θ for all nodes.
    """
    # Adjust image to ensure all values are non-negative
    image -= max(np.min(image), 0)
    im = np.exp(image)

    # Get pixel coordinates using higra embedding
    emb = hg.EmbeddingGrid2d(size)
    coord = emb.lin2grid(np.arange(tree.num_leaves()))

    # Initialize moment matrix with intensity weights
    m = np.zeros((tree.num_leaves(), 6), dtype=np.float64)
    m[:, 0] = 1                       # For total intensity (flux)
    m[:, 1] = coord[:, 0]              # x-coordinates
    m[:, 2] = coord[:, 1]              # y-coordinates
    m[:, 3] = coord[:, 0] ** 2         # x^2 for variance along x-axis
    m[:, 4] = coord[:, 1] ** 2         # y^2 for variance along y-axis
    m[:, 5] = coord[:, 0] * coord[:, 1]  # xy for covariance

    # Weight coordinates with intensities for each pixel in the node structure
    m = np.array([[im[int(i[1]), int(i[2])],
                   i[1] * im[int(i[1]), int(i[2])],
                   i[2] * im[int(i[1]), int(i[2])],
                   (i[1] ** 2) * im[int(i[1]), int(i[2])],
                   (i[2] ** 2) * im[int(i[1]), int(i[2])],
                   (i[1] * i[2]) * im[int(i[1]), int(i[2])]] for i in m])

    # Aggregate weighted moments over the tree structure
    m = hg.accumulate_sequential(tree, m, hg.Accumulators.sum)

    # First-order moments (centroid) for reference
    m00 = m[:, 0]   # Total intensity (flux sum)
    m10 = m[:, 1]   # Sum of weighted x-coordinates
    m01 = m[:, 2]   # Sum of weighted y-coordinates
    x_mean = m10 / m00
    y_mean = m01 / m00

    # Second-order moments
    x2 = (m[:, 3] / m00) - (x_mean ** 2)  # Variance along x
    y2 = (m[:, 4] / m00) - (y_mean ** 2)  # Variance along y
    xy = (m[:, 5] / m00) - (x_mean * y_mean)  # Covariance term

    # Calculate terms for major and minor axes
    lhs = (x2 + y2) / 2
    rhs = np.sqrt(np.power((x2 - y2) / 2, 2) + np.power(xy, 2))

    # Calculate major and minor axis lengths with non-negative constraint
    major_axis = np.sqrt(lhs + rhs)
    minor_axis = np.sqrt(np.maximum(lhs - rhs, 0))  # Ensure non-negative input to sqrt

    # Orientation angle θ
    t = np.arctan2(2 * xy, (x2 - y2))

    # Adjust angle θ based on quadrants (applies element-wise)
    theta = np.where((xy < 0) & (t > 0), (t - np.pi) / 2,
                     np.where((t < 0) & (xy > 0), (t + np.pi) / 2, t / 2))

    return major_axis, minor_axis, theta


def attribute_statistical_significance(tree, altitudes, volume, area, background_var, gain, alpha=1-1e-6):
    from scipy.stats import chi2

    volume /= (background_var + altitudes[tree.parents()] / gain)

    significant_nodes = volume > chi2.ppf(alpha, area)  # inverse cdf
    significant_nodes[:tree.num_leaves()] = False
    return significant_nodes


def attribute_main_branch(tree):
    # True is a node is not the root of a main branch
    area = hg.attribute_area(tree)
    largest_child = hg.accumulate_parallel(tree, area, hg.Accumulators.argmax)
    child_number = hg.attribute_child_number(tree)
    return child_number == largest_child[tree.parents()]


def select_objects(tree, significant_nodes):
    filtered_tree, node_map = hg.simplify_tree(tree, np.logical_not(significant_nodes))

    main_branch = attribute_main_branch(filtered_tree)

    # special case for the root
    if not significant_nodes[tree.root()]:
        root_children = filtered_tree.children(filtered_tree.root())
        main_branch[root_children] = False

    res = np.zeros(tree.num_vertices(), dtype=np.bool_)
    res[node_map] = np.logical_not(main_branch)
    return res


def move_up(tree, altitudes, area, objects, background_var, gain, gamma_distance, volume_ratio, gaussian, alambda=0.005):
    # true if a node is in the main branch of its parent
    main_branch = attribute_main_branch(tree)

    # index of the closest object ancestor in the main branch, itself otherwise
    closest_object_ancestor = hg.propagate_sequential(
        tree,
        np.arange(tree.num_vertices()),
        np.logical_and(main_branch, np.logical_not(objects)))

    # objects should be move up in the main branch to their target altitude
    target_altitudes = altitudes.copy()
    object_indexes, = np.nonzero(objects)
    local_noise = np.sqrt(background_var + altitudes[tree.parent(object_indexes)] / gain)
    target_altitudes[object_indexes] = altitudes[object_indexes] + alambda * local_noise

    # target altitude associated to the closest object anacestor in the main branch, self altitude otherwise
    target_altitudes = target_altitudes[closest_object_ancestor]
    # a move is valid if the target altitude of the closest object ancestor in the main branch is lower than the
    # the curent altitude and if the closest object is a real object
    #valid_moves = np.logical_and(
    #    np.logical_and(altitudes >= target_altitudes, objects[closest_object_ancestor]),
    #    np.logical_and(gamma_distance==1, altitudes/area>=gaussian),
    #)
    valid_moves = np.logical_and(
        np.logical_and(altitudes >= target_altitudes, objects[closest_object_ancestor]),
        altitudes/area>=gaussian,
    )

    parent_closest_object_ancestor = closest_object_ancestor[tree.parents()]
    parent_not_valid_moves = np.logical_not(valid_moves[tree.parents()])
    # a new object is a node n such that (valid_move(n) && (!valid_move(parent(n)) || closest(n) != closest(parent(n))))
    new_objects = np.logical_and(
        valid_moves,
        np.logical_or(
            parent_not_valid_moves,
            parent_closest_object_ancestor != closest_object_ancestor))
    return new_objects

