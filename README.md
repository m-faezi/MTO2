# MTO2 - Astronomical Source Detection Tool

##  Quick Start

> [!TIP]
> Before installing MTO2, it may be useful to create a python virtual environment (e.g. using [venv](https://docs.python.org/3/library/venv.html)) to avoid dependency version conflicts.
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