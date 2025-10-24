from datetime import datetime
from mto2lib.parser import make_parser
import os


class Run:

    def __init__(self):
        self.arguments = None
        self.results_dir = None
        self.status = None
        self.time_stamp = None

    def setup_args(self):

        self.status = "Running"
        self.arguments = make_parser().parse_args()
        self.time_stamp = datetime.now().isoformat()
        self.results_dir = os.path.join("./results", self.time_stamp)
        os.makedirs(self.results_dir, exist_ok=True)

        return self



