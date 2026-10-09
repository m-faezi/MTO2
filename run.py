from datetime import datetime
from mto2lib.parser import make_parser
import os
import shutil
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
        self.results_dir = os.path.join("./results", self.time_stamp)

        base = Path("./results").resolve()
        out_dir = Path(self.arguments.out_dir) if self.arguments.out_dir else None

        if out_dir is not None:
            target = (base / out_dir).resolve()

            if out_dir.is_absolute() or not target.is_relative_to(base):
                raise ValueError("--out-dir must be inside ./results")

            if target.exists():
                if not target.is_dir():
                    raise NotADirectoryError(target)
                shutil.rmtree(target)

            target.mkdir(parents=True, exist_ok=True)
            self.results_dir = str(target)

        return self



