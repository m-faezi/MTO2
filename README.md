# MTO
a tool for detecting sources in astronomical images

--------------------------

Build instructions:

python3 -m venv ./venvs/mto

source ./venvs/mto/bin/activate

pip install -r requirements.txt

--------------------------

To get help: 

	python mto.py -h

To run with default parameter: 

	python mto.py [input image path]

To include more faint outskirts of objects, a lower move_factor value is recommended: 

	python mto.py [input image path] -move_factor <float>

To extract and save parameters:

	python mto.py [input image path] -move_factor <float> --par_out

--------------------------

Arguments:

  -h, --help            Show the help message and exit

 -par_out		        Saves in .csv format
 
 -move_factor          Higher values reduce the spread of objects (default = 0.5)

-------------------------