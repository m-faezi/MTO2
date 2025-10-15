import mto2lib.main as mto2
from mto2lib import statistical_tests


class Run:

    def __init__(self):
        self.arguments = None
        self.results_dir = None
        self.status = None

    def setup_args(self):

        self.status = "Running"
        self.arguments, self.results_dir = mto2.setup_args()

        return self

    def detect_significant_objects(self, dark_frame, tree):

        significant_nodes = statistical_tests.attribute_statistical_significance(
            tree, dark_frame
        )

        tree.init_segments = statistical_tests.select_objects(tree, significant_nodes)

        return self

