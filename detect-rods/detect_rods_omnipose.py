"""Detect rod-shaped bacteria with Omnipose (deep-learning segmentation).

Omnipose is trained on rod-shaped bacteria and separates touching cells,
which thresholding cannot do. It needs its own environment (see README).

Reading, measurement and plotting are shared with detect_rods_skimage.py.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from cellpose_omni import models
from skimage import exposure, transform

from detect_rods_skimage import measure_rods, plot_rods, read_image

if __name__ == "__main__":
    # --- Parameters (change to fit your experiment) ---
    data_dir = Path(__file__).resolve().parent
    image_name = "image2.lsm"
    crop = (slice(150, -40), slice(50, -100))   # rows, columns to keep
    model_name = "bact_fluor_omni"   # fluorescence; "bact_phase_omni" for phase contrast
    upscale = 4              # Omnipose expects cells several px wide; ours are ~2 px
    mask_threshold = 0.0     # lower -> larger/more cells, higher -> fewer
    flow_threshold = 0.0     # 0 = no flow-error filtering (recommended by Omnipose)
    min_aspect_ratio = 2     # remove round objects
    extra_formats = ["pdf", "svg"]
    show_figures = True      # close the windows to end the script

    fig_dir = data_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    image_file = data_dir / image_name
    stem = image_file.stem + "_omnipose"

    # --- Read and upscale ---
    img, pixel_size = read_image(image_file)
    unit = "µm" if pixel_size else "px"
    pixel_size = pixel_size or 1.0
    img = exposure.rescale_intensity(img[crop], out_range=(0, 1))
    img_up = transform.rescale(img, upscale, order=3)
    print(f"Image: {image_file.name}, {img.shape} -> {img_up.shape}")

    # --- Segment ---
    use_gpu = torch.cuda.is_available()
    print("GPU:", use_gpu)
    if not use_gpu:
        # The pretrained models were saved on a GPU: force loading them on the CPU
        _torch_load = torch.load
        torch.load = lambda *args, **kwargs: _torch_load(*args, **{**kwargs, "map_location": "cpu"})
    model = models.CellposeModel(gpu=use_gpu, model_type=model_name)
    labels, flows, _ = model.eval(img_up, channels=None, omni=True,
                                  mask_threshold=mask_threshold,
                                  flow_threshold=flow_threshold)

    # --- Measure (pixel size of the upscaled image) ---
    px = pixel_size / upscale
    rods = measure_rods(labels, px)
    rods = rods[rods.axis_major_length / rods.axis_minor_length >= min_aspect_ratio]
    out_file = data_dir / f"{stem}_rods.csv"
    rods.to_csv(out_file, index=False)
    print(f"{len(rods)} rods saved to {out_file.name}")
    print(f"Median length: {rods.axis_major_length.median():.2f} {unit}, "
          f"median width: {rods.axis_minor_length.median():.2f} {unit}")

    # --- Figure: rods colored by orientation ---
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_rods(ax, labels, rods, px)
    ax.set_xlabel(unit)
    ax.set_ylabel(unit)
    for fmt in ["png"] + extra_formats:
        fig.savefig(fig_dir / f"{stem}_colorcrop.{fmt}", dpi=150, bbox_inches="tight")

    if show_figures:
        plt.show()
    plt.close("all")
    print("Done. Figures saved in", fig_dir)
