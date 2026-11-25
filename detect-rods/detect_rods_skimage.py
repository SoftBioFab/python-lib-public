"""Detect rod-shaped objects (e.g. bacteria) in a 2D image, with scikit-image.

Steps: read the image and its pixel size (tifffile), enhance thin elongated
objects with a ridge filter, threshold, clean up, measure each rod
(regionprops) in µm, and color each rod by its orientation.

Works at the original resolution (no upscaling).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile
from matplotlib.collections import PolyCollection
from skimage import exposure, filters, measure

# 180 colors, one per degree of orientation
HUSL_COLORS = np.array(sns.color_palette("husl", 180))


def read_image(path):
    """Read a 2D image and its pixel size in µm (None if not found)."""
    with tifffile.TiffFile(path) as tif:
        img = tif.asarray()
        pixel_size = None
        if tif.is_lsm:
            pixel_size = tif.lsm_metadata["VoxelSizeX"] * 1e6   # m -> µm
    img = np.squeeze(img)
    if img.ndim == 3:                       # color or multi-channel -> first channel
        img = img[..., 0] if img.shape[-1] <= 4 else img[0]
    return img.astype(float), pixel_size


def segment_rods(img, ridge_sigma=1.0, threshold=1.5, min_area=8, min_aspect_ratio=3):
    """Return a label image of the rods and the ridge-filtered image.

    ridge_sigma: about half the rod width, in pixels
    threshold: ridge threshold, in units of the Otsu threshold. Higher values
        separate touching rods better but lose dim ones.
    min_area: smallest object kept, in pixels
    min_aspect_ratio: smallest major/minor axis ratio kept
    """
    img = exposure.rescale_intensity(img, out_range=(0, 1))
    ridges = filters.sato(img, sigmas=[ridge_sigma], black_ridges=False)

    # Hysteresis threshold: strong ridges seed the objects, weaker ones extend them
    high = threshold * filters.threshold_otsu(ridges)
    mask = filters.apply_hysteresis_threshold(ridges, 0.8 * high, high)

    # Remove small and round objects
    labels = measure.label(mask, connectivity=1)   # no diagonal links between rods
    for r in measure.regionprops(labels):
        too_small = r.area < min_area
        too_round = (r.axis_minor_length == 0 or
                     r.axis_major_length / r.axis_minor_length < min_aspect_ratio)
        if too_small or too_round:
            labels[labels == r.label] = 0
    return measure.label(labels > 0, connectivity=1), ridges


def measure_rods(labels, pixel_size=1.0):
    """Table of rod properties. Lengths in µm (or px if pixel_size=1).

    orientation_deg: angle of the major axis from the x axis, 0-180°,
    counterclockwise as seen on the displayed image.
    """
    props = measure.regionprops_table(
        labels, properties=("label", "centroid", "axis_major_length",
                            "axis_minor_length", "orientation", "area", "solidity"))
    df = pd.DataFrame(props).rename(columns={"centroid-0": "y", "centroid-1": "x"})
    # skimage orientation: angle from the row axis -> convert to angle from x axis
    df["orientation_deg"] = (np.degrees(df.pop("orientation")) + 90) % 180
    for col in ["x", "y", "axis_major_length", "axis_minor_length"]:
        df[col] *= pixel_size
    df["area"] *= pixel_size**2
    return df


def plot_rods(ax, labels, rods, pixel_size=1.0):
    """Draw each rod in the `rods` table as a filled outline colored by
    orientation (vector shapes)."""
    angles = dict(zip(rods["label"], rods["orientation_deg"]))
    shapes, colors = [], []
    for r in measure.regionprops(labels):
        if r.label not in angles:
            continue
        angle = angles[r.label]
        padded = np.pad(r.image, 1)
        y0, x0 = r.bbox[0] - 1, r.bbox[1] - 1
        for c in measure.find_contours(padded.astype(float), 0.5):
            shapes.append((c[:, ::-1] + [x0, y0]) * pixel_size)   # (x, y)
            colors.append(HUSL_COLORS[int(angle) % 180])
    ax.add_collection(PolyCollection(shapes, facecolors=colors, edgecolors="none"))
    h, w = labels.shape
    ax.set_xlim(0, w * pixel_size)
    ax.set_ylim(h * pixel_size, 0)          # image orientation (y down)
    ax.set_aspect("equal")
    ax.set_facecolor("black")


if __name__ == "__main__":
    # --- Parameters (change to fit your experiment) ---
    data_dir = Path(__file__).resolve().parent
    image_name = "image2.lsm"
    crop = (slice(150, -40), slice(50, -100))   # rows, columns to keep
    ridge_sigma = 1.0                           # ~ half the rod width (px)
    threshold = 1.5                             # x Otsu; higher separates touching rods better
    min_area = 8                                # px
    min_aspect_ratio = 3
    extra_formats = ["pdf", "svg"]              # vector versions of the rod image
    show_figures = True                         # close the windows to end the script

    fig_dir = data_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    image_file = data_dir / image_name
    stem = image_file.stem + "_skimage"

    # --- Read ---
    img, pixel_size = read_image(image_file)
    if pixel_size is None:
        pixel_size, unit = 1.0, "px"
    else:
        unit = "µm"
    img = img[crop]
    print(f"Image: {image_file.name}, {img.shape}, pixel size: {pixel_size:.3f} {unit}")

    # --- Segment and measure ---
    labels, ridges = segment_rods(img, ridge_sigma, threshold, min_area, min_aspect_ratio)
    rods = measure_rods(labels, pixel_size)
    out_file = data_dir / f"{stem}_rods.csv"
    rods.to_csv(out_file, index=False)
    print(f"{len(rods)} rods saved to {out_file.name}")
    print(f"Median length: {rods.axis_major_length.median():.2f} {unit}, "
          f"median width: {rods.axis_minor_length.median():.2f} {unit}")

    # --- Figures ---
    extent = (0, img.shape[1] * pixel_size, img.shape[0] * pixel_size, 0)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, im, title in zip(axs, [img, ridges, labels > 0],
                             ["Original", "Ridge filter", "Rods"]):
        ax.imshow(im, cmap="gray", extent=extent)
        ax.set_title(title)
        ax.set_xlabel(unit)
    fig.savefig(fig_dir / f"{stem}_steps.png", dpi=150, bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(6, 6))
    plot_rods(ax, labels, rods, pixel_size)
    ax.set_xlabel(unit)
    ax.set_ylabel(unit)
    for fmt in ["png"] + extra_formats:
        fig.savefig(fig_dir / f"{stem}_colorcrop.{fmt}", dpi=150, bbox_inches="tight")

    fig, ax = plt.subplots()
    ax.hist(rods.orientation_deg, bins=36, range=(0, 180))
    ax.set_xlabel("Orientation (deg)")
    ax.set_ylabel("Number of rods")
    fig.savefig(fig_dir / f"{stem}_orientation_hist.png", dpi=150, bbox_inches="tight")

    if show_figures:
        plt.show()
    plt.close("all")
    print("Done. Figures saved in", fig_dir)
