from image import Image
from dark_frame import DarkFrame
from mto2_run import MTO2Run
from max_tree import MaxTree
from extractor import Extractor

from mto2lib.utils import io_utils
from mto2lib import statistical_tests
import sys


def execute_run():

    run = MTO2Run()
    image = Image()
    dark_frame = DarkFrame()

    try:

        run.setup_args()
        image.get_image(run.arguments)
        image.preprocess_image(run.arguments.s_sigma)

        if run.arguments.background_mode == 'const':

            try:

                dark_frame.estimate_const_bg(image.smooth_image)
                dark_frame.create_reduced_image(image, run.results_dir)

                maxtree = MaxTree()
                maxtree.construct_max_tree(image.reduced_image)
                maxtree.compute_attributes(run, image)

            except Exception as e:

                run.arguments.background_mode = 'morph'

                print(f"Note: Background mode switched from 'const' to '{run.arguments.background_mode}'!")

                maxtree = MaxTree()
                maxtree.construct_max_tree(image.smooth_image)
                maxtree.compute_attributes(run, image)
                dark_frame.estimate_morph_bg(image, maxtree)

        else:

            maxtree = MaxTree()
            maxtree.construct_max_tree(image.smooth_image)
            maxtree.compute_attributes(run, image)
            dark_frame.estimate_morph_bg(image, maxtree)

        io_utils.save_run_metadata(run)
        dark_frame.save_background(run.results_dir, image.header, run.arguments)
        dark_frame.create_reduced_image(image, run.results_dir)

        run.detect_significant_objects(dark_frame, maxtree)
        modified_isophote = statistical_tests.move_up(maxtree, dark_frame, run)

        extractor = Extractor()
        extractor.create_segmentation(maxtree, image, modified_isophote, run)

        if run.arguments.par_out:

            extractor.extract_parameters(extractor, maxtree, run, image)

        run.status = "Completed"
        io_utils.save_run_metadata(run)

        print("MTO2 run completed successfully!")

    except KeyboardInterrupt:

        run.status = "Interrupted"
        io_utils.save_run_metadata(run)

        print("\nMTO2 run interrupted by user!")
        sys.exit(1)

    except Exception as e:

        run.status = "Terminated"
        io_utils.save_run_metadata(run)

        print(f"MTO2 run terminated with error: {e}")

        sys.exit(1)


if __name__ == "__main__":

    execute_run()

