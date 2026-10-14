from datetime import datetime
from mto2lib.parser import make_parser
import os
from pathlib import Path


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

        if self.arguments.out_dir:
            self.results_dir = os.path.join("./results", self.arguments.out_dir)

            if Path(self.results_dir).is_absolute():
                raise ValueError("--out-dir is absolute")

            if Path(self.results_dir).exists():
                if not Path(self.results_dir).is_dir():
                    raise NotADirectoryError(Path(self.results_dir))

            Path(self.results_dir).mkdir(parents=True, exist_ok=True)

        else:

            self.results_dir = os.path.join("./results", self.time_stamp)
            Path(self.results_dir).mkdir(parents=True, exist_ok=True)

        return self



