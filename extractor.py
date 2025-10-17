from mto2lib import segment, parameter_extraction


class Extractor:

    def __init__(self):
        self.maxtree_of_segment = None
        self.segment_node_map = None
        self.ids = None

    def create_segmentation(self, tree, image, run):

        self.maxtree_of_segment, self.segment_node_map, self.ids = segment.get_segmentation_map(
            tree.tree_structure,
            tree.corrected_segments,
            image.header,
            run.arguments
        )

        return self

    def extract_parameters(self, extractor, maxtree, run, image):

        parameter_extraction.extract_parameters(
            image.image,
            image.header,
            extractor.maxtree_of_segment,
            extractor.segment_node_map,
            maxtree.parent_altitude,
            maxtree.area,
            extractor.ids,
            run.arguments,
        )

        return self

