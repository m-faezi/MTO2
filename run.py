import mto2lib.main as mto2


class Run:

    def __init__(self):
        self.arguments = None
        self.results_dir = None
        self.status = None

    def setup_args(self):

        self.status = "Running"
        self.arguments, self.results_dir = mto2.setup_args()

        return self



