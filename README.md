# MTO
## a tool for detecting sources in astronomical images

--------------------------

#### Build instructions:

    $ python3 -m venv ./venvs/mto

    $ source ./venvs/mto/bin/activate

    $ pip install -r requirements.txt

--------------------------

#### To get help: 

	$ python mto.py -h

#### To run with default parameter: 

	$ python mto.py [input image path]

#### To include more faint outskirts of objects, a lower move_factor value is recommended: 

	$ python mto.py [input image path] --move_factor <float>

#### For deblending correction: 

	$ python mto.py [input image path] --area_ratio <float>

#### To extract and save parameters:

	$ python mto.py [input image path] --move_factor <float> --par_out

#### To apply a morphological Gaussian filter:

	$ python mto.py [input image path] --move_factor <float> --G_fit

#### To get background subtracted image:

	$ python mto.py [input image path] --move_factor <float> --reduce

#### Optional string to append to output file names:

	$ python mto.py [input image path] --move_factor <float> --file_tag <string>

#### To specify a directory for output files:

	$ python mto.py [input image path] --move_factor <float> --output_path <directory>

--------------------------

#### Command line arguments:

```yaml
-h, --help            Show the help message and exit
--move_factor         Adjust the spread of objects. Type: float Default: 0 Range: non-negative
--area_ratio          Adjust deblending correction. Type: float Default: 0.78 Range: [0.0, 1.0]
--par_out             Extract and save parameters in .csv format
--G_fit               Applies morphological Gaussian filter
--reduce              Return background subtracted image
--file_tag            Optional string to append to output file names
--output_path         Directory to save output files (default = current directory)

