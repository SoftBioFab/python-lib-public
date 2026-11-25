# Detect rods

Detects rod-shaped objects (e.g. bacteria) in a 2D microscopy image, colors them by orientation and exports their properties. Three versions:

| Script | Method | Notes |
|---|---|---|
| `detect_rods.py` | Band-pass filter + adaptive threshold (OpenCV) | Original method; image upscaled 4×; output in px |
| `detect_rods_skimage.py` | Ridge filter (`sato`) + hysteresis threshold (scikit-image) | Original resolution, ~6× faster; reads pixel size, output in µm |
| `detect_rods_omnipose.py` | Omnipose deep-learning segmentation | Separates touching rods; needs its own environment |

`detect_rods_notebook.ipynb` is the original notebook version of `detect_rods.py`.

An example image, `image2.lsm`, is included. Parameters are at the start of the main part of each script.

## Usage

```
python detect_rods_skimage.py
```

Figures are saved in `figures/`, rod properties in a `.csv` (or `.dat`) file next to the script.

Orientation convention (`detect_rods_skimage.py`, `detect_rods_omnipose.py`): angle of the rod from the x axis, 0–180°, counterclockwise on the displayed image.

## Omnipose environment

Omnipose (with PyTorch) is large, so keep it in a separate environment:

```
conda create -n omnipose python=3.10
conda activate omnipose
pip install omnipose tifffile seaborn pandas
python detect_rods_omnipose.py
```

The model is downloaded on first use. See the [Omnipose documentation](https://omnipose.readthedocs.io) for GPU support.
