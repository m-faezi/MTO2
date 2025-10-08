import higra as hg
import numpy as np
from PIL import Image, ImageOps
import mto2lib.io_utils as io_uts
import os


def get_segmentation_map(tree_structure, modified_isophote, header, arguments):

    tree_of_segments, n_map_segments = hg.simplify_tree(tree_structure, np.logical_not(modified_isophote))
    colors = np.random.randint(0, 254, (tree_of_segments.num_vertices(), 3), dtype=np.uint8)
    colors[tree_of_segments.root(), :] = 0
    seg = hg.reconstruct_leaf_data(tree_of_segments, colors)

    segmentation_image = Image.fromarray(seg.astype(np.uint8))
    segmentation_image = ImageOps.flip(segmentation_image)
    unique_segment_ids = np.arange(tree_of_segments.num_vertices())[::-1]
    seg_with_ids = hg.reconstruct_leaf_data(tree_of_segments, unique_segment_ids)

    output_png = os.path.join(arguments.output_path, "segmentation_map.png")
    output_fits = os.path.join(arguments.output_path, "segmentation_map.fits")

    segmentation_image.save(output_png, 'PNG', quality=1080)
    io_uts.save_fits_with_header(seg_with_ids, header, output_fits)

    return tree_of_segments, n_map_segments, unique_segment_ids

