# MTO2 - Astronomical Source Detection Tool

[![DOI](https://img.shields.io/badge/DOI-10.1515/mathm-blue.svg)](https://doi.org/10.1515/mathm-2016-0006)
[![Astropy](https://img.shields.io/badge/powered%20by-Astropy-orange.svg)](https://www.astropy.org/)
[![Higra](https://img.shields.io/badge/Powered%20by-Higra-green.svg)](https://higra.readthedocs.io/)
##  Quick Start

> [!TIP]
> Before installing MTO2, it may be useful to create a python virtual environment (e.g. using [venv](https://docs.python.org/3/library/venv.html)) to avoid dependency version conflicts.

### Dependencies

The dependencies are listed in the [requirements.txt](requirements.txt) file.

```bash
python3 -m venv ./venvs/mto2
source ./venvs/mto2/bin/activate
pip install -r requirements.txt
```
### Minimal run
```bash
python mto2.py image.fits
```

### Get help
```bash
python mto2.py -h
```

### Tuned run
```bash
python mto2.py image.fits
    --s_sigma 1.6 
    --move_factor 0.1 
    --area_ratio 0.91  
    --G_fit 
    --reduce 
    --par_out 
    --output_path "./results" 
    --file_tag "tuned_run"
```
### Command line arguments

| Option            | Description                                   | Type     | Default | Range/Values |
|-------------------|-----------------------------------------------|----------|---------|--------------|
| `-h`, `--help`    | Show the help message and exit                | flag     | -       | -            |
| `--move_factor`   | Adjust the spread of objects                  | float    | 0.00    | ≥ 0          |
| `--area_ratio`    | Adjust deblending sensitivity                 | float    | 0.90    | [0.0, 1.0)   |
| `--par_out`       | Extract and save parameters in .csv format    | flag     | -       | -            |
| `--G_fit`         | Apply Gaussian-fit attribute filter           | flag     | -       | -            |
| `--reduce`        | Return background-subtracted image            | flag     | -       | -            |
| `--s_sigma`       | Standard deviation for smoothing kernel       | float    | 2.00    | ≥ 0          |
| `--file_tag`      | Optional string to append to output filenames | string   | ""      | Any text     |
| `--output_path`   | Directory to save output files                | string   | "."     | Valid path   |


## Acknowledgments

This software was developed for a Ph.D. thesis (Mohammad H. Faezi, [2025](#4)) at the Rijksuniversiteit of Groningen under the supervision of Prof. Dr. Reynier Peletier and Dr. Michael Wilkinson.

MTO2 is developed using the [Higra](https://github.com/higra/Higra) Python package, and builds on their example implementation of MTO: [Astronomical object detection with the Max-Tree - MMTA 2016](https://higra.readthedocs.io/en/stable/notebooks.html#illustrative-applications-from-scientific-papers).


## Bibliography

- <a id="1">Teeninga P., Moschini U., Trager S. C., et al. (2016). “Statistical attribute filtering to detect faint extended astronomical sources”. In: *Mathematical Morphology &mdash; Theory and Applications* 1.1. DOI: [10.1515/mathm-2016-0006](https://doi.org/10.1515/mathm-2016-0006).</a>
- <a id="5">Faezi M. H., Peletier R., & Wilkinson M. H. (2024). “Multi-Spectral Source-Segmentation Using Semantically-Informed Max-Trees”. In: *IEEE Access* 12, pp. 72288 - 72302. DOI: [10.1109/ACCESS.2024.3403309](https://doi.org/10.1109/ACCESS.2024.3403309).</a>
- <a id="5">Salembier P., Oliveras A., & Garrido L. (1998). “Antiextensive connected operators for image and sequence processing”. In: *IEEE Transactions on Image Processing* 7.4, pp. 555–570. DOI: [10.1109/83.663500](https://doi.org/10.1109/83.663500).</a>
