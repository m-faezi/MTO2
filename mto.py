import higra as hg
import numpy as np

import background
import helper
from PIL import Image, ImageOps
from matplotlib import pyplot as plt


data_path = 'data/crop_ngc4307_g.fits'
#data_path = '../whale/data/bg_sim/intensity variation 1/cluster1.fits'
image, header = helper.read_image_data(data_path, 0, -1, 0, -1)
image = helper.image_value_check(image)
image = helper.smooth_filter(image)

bg_mean, bg_var, bg_gain = background.estimate_background(image)

image_calibrated = image - bg_mean

graph_structure, tree_structure, altitudes = helper.image_to_hierarchical_structure(image_calibrated)

area = hg.attribute_area(tree_structure)
volume = hg.attribute_volume(tree_structure, altitudes)
parent_volume = volume[tree_structure.parents()]
gamma = hg.attribute_topological_height(tree_structure)
parent_gamma = gamma[tree_structure.parents()]

significant_nodes = helper.attribute_statistical_significance(
    tree_structure,
    altitudes,
    volume,
    area,
    bg_var,
    bg_gain,
)

objs = helper.select_objects(tree_structure, significant_nodes)

nobjs = helper.move_up(tree_structure, altitudes, objs, bg_var, bg_gain, parent_gamma-gamma, volume/parent_volume)

# construct final segmentation with random colors as labels
colors = np.random.randint(0, 256, (tree_structure.num_vertices(), 3), dtype=np.uint8)
colors[tree_structure.root(),:] = 0
seg = hg.reconstruct_leaf_data(tree_structure, colors, np.logical_not(nobjs))

segmentation_image = Image.fromarray(seg.astype(np.uint8))
segmentation_image = ImageOps.flip(segmentation_image)
segmentation_image.save('MTO-detection.png', 'PNG', quality=95)

