# Python port (2x first)

This folder contains a functional (not optimized) Python port of the MATLAB code in the repo.

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run 2x demo

From `python/`:

```bash
python run_2x.py --input ..\TESTSET\1_elk.bmp --outdir outputs
```

It prints PSNR/SSIM and writes `outputs/<name>_2x.png`.

## Deployment-style usage (API + CLI)

### CLI

```bash
python cli_2x.py --input ..\TESTSET\ceshi_1.jpg --outdir outputs --crop 64
```

### Python API

```python
import imageio.v3 as iio
from siisls import interpolate_2x_array

img = iio.imread(r"..\TESTSET\ceshi_1.jpg")
y2x = interpolate_2x_array(img, crop=64)  # float64, [0,255]
```

