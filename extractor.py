from mto2lib import segment, parameter_extraction


class Extractor:

    def __init__(self):
        self.maxtree_of_segment = None
        self.segment_node_map = None
        self.ids = None
        self.parent_segment_ids = None

    def create_segmentation(self, tree, image, run):

        self.maxtree_of_segment, self.segment_node_map, self.ids, self.parent_segment_ids = segment.get_segmentation_map(
            tree.tree_structure,
            tree.corrected_segments,
            image.header,
            run
        )

        return self

    def extract_parameters(self, extractor, maxtree, run, image):

        parameter_extraction.extract_parameters(
            image.image,
            image.header,
            extractor.maxtree_of_segment,
            extractor.segment_node_map,
            maxtree.altitudes,
            maxtree.area,
            maxtree.variance,
            maxtree.convexness,
            extractor.ids,
            extractor.parent_segment_ids,
            run
        )

        return self

