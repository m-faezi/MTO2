# MTO - Astronomical Source Detection Tool

##  Quick Start

```bash
python3 -m venv ./venvs/mto
source ./venvs/mto/bin/activate
pip install -r requirements.txt
```
### Run basic detection
```bash
python mto.py image.fits
```

### Show help
```bash
python mto.py -h
```

### Full processing
```bash
python mto.py image.fits
    --s_sigma 1.6 
    --move_factor 0.1 
    --area_ratio 0.91  
    --G_fit 
    --reduce 
    --par_out 
    --output_path "./results" 
    --file_tag "science_run"
```
### Command line arguments

| Option            | Description                                   | Type     | Default | Range/Values |
|-------------------|-----------------------------------------------|----------|---------|--------------|
| `-h`, `--help`    | Show the help message and exit                | flag     | -       | -            |
| `--move_factor`   | Adjust the spread of objects                  | float    | 0.00    | ≥ 0          |
| `--area_ratio`    | Adjust deblending sensitivity                 | float    | 0.90    | [0.0, 1.0)   |
| `--par_out`       | Extract and save parameters in .csv format    | flag     | -       | -            |
| `--G_fit`         | Apply morphological Gaussian filter           | flag     | -       | -            |
| `--reduce`        | Return background-subtracted image            | flag     | -       | -            |
| `--s_sigma`       | Standard deviation for smoothing kernel       | float    | 2.00    | ≥ 0          |
| `--file_tag`      | Optional string to append to output filenames | string   | ""      | Any text     |
| `--output_path`   | Directory to save output files                | string   | "."     | Valid path   |