from image import Image
from dark_frame import DarkFrame
from run import Run
from max_tree import MaxTree
from extractor import Extractor
import numpy as np
from mto2lib.utils import io_utils
import sys


def execute_run():

    run = Run()
    image = Image()
    dark_frame = DarkFrame()

    try:

        run.setup_args()
        image.get_image(run)
        image.preprocess_image(run.arguments.s_sigma)

        if run.arguments.background_mode == 'const':

            try:

                dark_frame.estimate_const_bg(image.smooth_image)

                if run.arguments.skip_reduction:

                    dark_frame.bg_map = np.full_like(image.image, 0, dtype=np.float32)

                dark_frame.create_reduced_image(image, run.results_dir)

                maxtree = MaxTree()
                maxtree.construct_max_tree(image.smooth_reduced_image)
                maxtree.compute_attributes(run, image)

            except Exception as e:

                maxtree = MaxTree()
                maxtree.construct_max_tree(image.smooth_image)
                maxtree.compute_attributes(run, image)
                dark_frame.estimate_morph_bg(image, maxtree)

                if run.arguments.skip_reduction:

                    dark_frame.bg_map = np.full_like(image.image, 0, dtype=np.float32)

                else:

                    dark_frame.bg_map = np.full_like(image.image, dark_frame.bg_mean, dtype=np.float32)

                dark_frame.create_reduced_image(image, run.results_dir)

        else:

            maxtree = MaxTree()
            maxtree.construct_max_tree(image.smooth_image)
            maxtree.compute_attributes(run, image)
            dark_frame.estimate_morph_bg(image, maxtree)

            if run.arguments.skip_reduction:

                dark_frame.bg_map = np.full_like(image.image, 0, dtype=np.float32)

            dark_frame.create_reduced_image(image, run.results_dir)

        io_utils.save_run_metadata(run)
        dark_frame.save_background(run.results_dir, image.header, run.arguments)

        maxtree.detect_significant_objects(dark_frame, run)
        maxtree.move_up(dark_frame, run)

        extractor = Extractor()
        extractor.create_segmentation(maxtree, image, run)

        dark_frame.create_cali_base(
            extractor.maxtree_of_segment,
            extractor.segment_node_map,
            maxtree,
            run.results_dir,
            image.header
        )

        if run.arguments.par_out:

            extractor.extract_parameters(extractor, maxtree, run, image)

        run.status = "Completed"
        io_utils.save_run_metadata(run)

        print("\nMTO2 run completed successfully!")

    except KeyboardInterrupt:

        run.status = "Interrupted"
        io_utils.save_run_metadata(run)

        print("\nMTO2 run interrupted by user!")
        sys.exit(1)

    except Exception as e:

        run.status = "Terminated"
        io_utils.save_run_metadata(run)

        print(f"\nMTO2 run terminated with error: {e}")

        sys.exit(1)


if __name__ == "__main__":

    execute_run()

