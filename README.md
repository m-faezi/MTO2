# MTO2 - Astronomical Source Detection Tool

[![DOI](https://img.shields.io/badge/DOI-10.1515/mathm-blue.svg)](https://doi.org/10.1515/mathm-2016-0006)
[![Higra](https://img.shields.io/badge/Powered%20by-Higra-green.svg)](https://higra.readthedocs.io/)
[![Astropy](https://img.shields.io/badge/powered%20by-Astropy-orange.svg)](https://www.astropy.org/)
<a href="https://github.com/m-faezi/MTO2/blob/main/CONTRIBUTING.md" alt="contributions welcome"><img src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg"/></a>

##  Quick Start

> [!TIP]
> It is recommended to use an isolated Python virtual environment to avoid dependency conflicts. The simplest way is to use Python's built-in [venv](https://docs.python.org/3/library/venv.html). If you already have another virtual environment active, be sure to deactivate it first.



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
    --crop "10, 10000, 20, 20000"
```

Get started with a demo in Google Colab:
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1yjNcUJwqliQEY0N7AYLkUD2QrubqkAzc?usp=sharing)


### Command line arguments

<small>

| Option          | Description                                   | Type   | Default                      | Range/Values |
|-----------------|-----------------------------------------------|--------|------------------------------|--------------|
| `--s_sigma`     | Standard deviation for smoothing kernel       | float  | 2.00                         | ≥ 0          |
| `--move_factor` | Adjust the spread of objects                  | float  | 0.00                         | ≥ 0          |
| `--area_ratio`  | Adjust deblending sensitivity                 | float  | 0.90                         | [0.0, 1.0)   |
| `--output_path` | Directory to save output files                | string | "."                          | valid path   |
| `--file_tag`    | Optional string to append to output filenames | string | ""                           | any text     |
| `--par_out`     | Extract and save parameters in .csv format    | flag   | -                            | -            |
| `--reduce`      | Return background-subtracted image            | flag   | -                            | -            |
| `--G_fit`       | Apply Gaussian-fit attribute filter           | flag   | -                            | -            |
| `--crop`        | Crops the image                               | string | "x_min, y_min, x_max, y_max" | image domain |
| `-h`, `--help`  | Show the help message and exit                | flag   | -                            | -            |

</small>

## Acknowledgments

This software was developed for **Faint Object Detection in Multidimensional Astronomical Data** Ph.D. thesis (Mohammad H. Faezi, [2025](#4)) at the Rijksuniversiteit of Groningen under the supervision of Dr. Michael Wilkinson and Prof. Dr. Reynier Peletier.

MTO2 is developed using the [Higra](https://github.com/higra/Higra) Python package, and builds on their example implementation of MTO: [Astronomical object detection with the Max-Tree - MMTA 2016](https://higra.readthedocs.io/en/stable/notebooks.html#illustrative-applications-from-scientific-papers).

This implementation draws inspiration from [Caroline Haigh's work](https://github.com/CarolineHaigh/mtobjects).

## Bibliography

- <a id="1">Teeninga P., Moschini U., Trager S. C., et al. (2016). “Statistical attribute filtering to detect faint extended astronomical sources”. In: *Mathematical Morphology &mdash; Theory and Applications* 1.1. DOI: [10.1515/mathm-2016-0006](https://doi.org/10.1515/mathm-2016-0006).</a>
- <a id="2">Haigh C., Chamba N., Vanhola A., et al. (2021). “Optimising and comparing source-extraction tools using objective segmentation quality criteria”. In: *Astronomy & Astrophysics* 645. DOI: [10.1051/0004-6361/201936561](https://doi.org/10.1051/0004-6361/201936561).</a>
- <a id="3">Faezi M. H., Peletier R., & Wilkinson M. H. (2024). “Multi-Spectral Source-Segmentation Using Semantically-Informed Max-Trees”. In: *IEEE Access* 12, pp. 72288 - 72302. DOI: [10.1109/ACCESS.2024.3403309](https://doi.org/10.1109/ACCESS.2024.3403309).</a>
- <a id="4">Salembier P., Oliveras A., & Garrido L. (1998). “Antiextensive connected operators for image and sequence processing”. In: *IEEE Transactions on Image Processing* 7.4, pp. 555–570. DOI: [10.1109/83.663500](https://doi.org/10.1109/83.663500).</a>

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

