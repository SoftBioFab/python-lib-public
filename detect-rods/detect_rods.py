"""Detect rod-shaped objects (e.g. bacteria) in a 2D image.

Steps: crop and upscale the image, band-pass filter, adaptive threshold,
remove small and low-aspect-ratio objects, color each rod by its
orientation, and export position, size and orientation of each rod.

The functions can also be imported from a notebook:
    from detect_rods import process_image, color_image, regionprops
"""
import re
from pathlib import Path

import cv2 as cv
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage
import seaborn as sns
import tifffile
from scipy.signal import windows

# 180 colors, one per degree of orientation
HUSL_COLORS = sns.color_palette("husl", 180)


# ---------------------------------------------------------------------------
# Image filtering
# ---------------------------------------------------------------------------
def mat2gray(img, scale=1):
    """Rescale an image to the range [0, scale]."""
    img = img - img.min()
    return img * np.double(scale) / img.max()


def unsharp(img, sigma=5, amount=10):
    """Unsharp mask. With amount=0, a plain Gaussian blur of width sigma."""
    if not sigma:
        return img
    img = np.double(img)
    blurred = cv.GaussianBlur(img, (0, 0), np.double(sigma))  # kernel size from sigma
    if amount:
        return img * (1 + amount) - blurred * amount
    return blurred


def _normalize(x):
    """Normalize an array so that it sums to 1."""
    return x / np.sum(x)


def bpass(img, lnoise=1, lobject=8):
    """Band-pass filter: remove noise of size lnoise and background larger
    than lobject. Translated from bpass.m by John C. Crocker and David G. Grier.
    """
    lnoise = np.double(lnoise)
    lobject = int(lobject)
    image = np.double(img)

    gauss_kernel = _normalize(windows.gaussian(int(np.floor(10 * lnoise)) + 1, lnoise))
    gconv = cv.filter2D(np.transpose(cv.filter2D(np.transpose(image), -1, gauss_kernel)),
                        -1, gauss_kernel)
    if lobject > 0:
        boxcar_kernel = _normalize(windows.boxcar(2 * lobject))
        bconv = cv.filter2D(np.transpose(cv.filter2D(np.transpose(image), -1, boxcar_kernel)),
                            -1, boxcar_kernel)
        filtered = gconv - bconv
        filtered[filtered < 0] = 0
    else:
        filtered = gconv.copy()
    return filtered


def fill_holes(img):
    """Fill holes in a binary image."""
    return ndimage.binary_fill_holes(img).astype(int)


# ---------------------------------------------------------------------------
# Object selection
# ---------------------------------------------------------------------------
def _find_contours(bw_img):
    contours, _ = cv.findContours(np.uint8(bw_img), mode=cv.RETR_TREE,
                                  method=cv.CHAIN_APPROX_SIMPLE)
    return contours


def _find_axes(m):
    """Major and minor axis of the ellipse having moments m."""
    if m["m00"] <= 0:
        return 1, 1
    a = m["mu20"] + m["mu02"]
    b = ((m["mu20"] - m["mu02"]) ** 2 + 4 * m["mu11"] ** 2) ** 0.5
    major = 2 * (2 * (a + b) / m["m00"]) ** 0.5
    minor = 2 * (2 * (a - b) / m["m00"]) ** 0.5
    return major, minor


def remove_small_blobs(bw_img, min_area=10):
    """Remove objects with area <= min_area from a binary image."""
    bw_img = np.uint8(bw_img)
    mask = np.zeros(bw_img.shape)
    contours = _find_contours(bw_img)
    for i, c in enumerate(contours):
        if cv.contourArea(c) <= min_area:
            cv.drawContours(mask, contours, i, 1, thickness=-1)
    return np.uint8(bw_img * (1 - mask))


def remove_small_aspect_ratio(bw_img, min_aspect_ratio=4):
    """Remove objects with major/minor axis ratio <= min_aspect_ratio."""
    bw_img = np.uint8(bw_img)
    mask = np.zeros(bw_img.shape)
    contours = _find_contours(bw_img)
    for i, c in enumerate(contours):
        if len(c) > 5:
            major, minor = _find_axes(cv.moments(c))
            if major / minor <= min_aspect_ratio:
                cv.drawContours(mask, contours, i, 1, thickness=-1)
    return np.uint8(bw_img * (1 - mask))


def process_image(img_in, s_blur=0.5, s_amount=0, lnoise=1.5, lobject=6,
                  box_size=13, small_blob_size=120, normalize=True,
                  fill=True, min_aspect_ratio=3, threshold_offset=0):
    """Binarize a grayscale image and keep only large, elongated objects.

    s_blur: size of the noise removed by a Gaussian blur
    s_amount: strength of the unsharp mask (0 = blur only)
    lnoise: typical noise size for the band-pass filter
    lobject: typical object size for the band-pass filter
    box_size: box size for the adaptive threshold (odd number)
    threshold_offset: pixels must be this much brighter than their local
        mean to be kept (0-255 scale); > 0 removes background noise
    """
    img = mat2gray(img_in, 255)
    img_u = unsharp(img, s_blur, s_amount)
    img_b = unsharp(bpass(img_u, lnoise, lobject), lnoise, 0)
    if normalize:
        img_b = img_b / np.max(img_b) * 255
    bw = cv.adaptiveThreshold(np.uint8(img_b), 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv.THRESH_BINARY, int(box_size), -threshold_offset)
    if fill:
        bw = fill_holes(bw)
    bw = remove_small_blobs(bw, small_blob_size / 2)
    bw = remove_small_aspect_ratio(bw, min_aspect_ratio)
    return bw


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------
def color_image(bw_img, polygon=((0, 0), (0, 1024), (1024, 1024), (1024, 0))):
    """Color each object inside `polygon` by its orientation.

    Returns a color image and a binary image of the kept objects.
    """
    bw_img = np.uint8(bw_img)
    bw = np.zeros(bw_img.shape)
    color = np.zeros(bw_img.shape + (3,))
    region = mpath.Path(polygon)
    contours = _find_contours(bw_img)
    for i, c in enumerate(contours):
        if len(c) > 5:
            centroid, _, angle = cv.fitEllipse(c)
            if region.contains_point(centroid):
                rgb = np.array(HUSL_COLORS[int(angle) % 180]) * 255
                cv.drawContours(color, contours, i, color=rgb, thickness=-1)
                cv.drawContours(bw, contours, i, color=255, thickness=-1)
    return np.uint8(color), np.uint8(bw)


def plot_rods_vector(ax, bw_img, polygon=((0, 0), (0, 1024), (1024, 1024), (1024, 0))):
    """Draw each object inside `polygon` as a filled outline colored by
    orientation. Vector shapes: stay sharp at any zoom in SVG/PDF."""
    from matplotlib.collections import PolyCollection
    region = mpath.Path(polygon)
    shapes, colors = [], []
    for c in _find_contours(bw_img):
        if len(c) > 5:
            centroid, _, angle = cv.fitEllipse(c)
            if region.contains_point(centroid):
                shapes.append(c[:, 0, :])
                colors.append(HUSL_COLORS[int(angle) % 180])
    ax.add_collection(PolyCollection(shapes, facecolors=colors, edgecolors="none"))
    h, w = bw_img.shape
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)                   # image orientation (y down)
    ax.set_aspect("equal")
    ax.set_facecolor("black")


def regionprops(bw_img, scale=1):
    """Properties of each object, similar to MATLAB's regionprops.

    Returns an array with one row per object:
    [x, y, major axis, minor axis, orientation (deg), area, solidity]
    """
    rows = []
    for c in _find_contours(bw_img):
        if len(c) < 5:
            continue
        _, _, angle = cv.fitEllipse(c)
        m = cv.moments(c)
        if m["m00"] > 0:
            major, minor = _find_axes(m)
            x, y = m["m10"] / m["m00"], m["m01"] / m["m00"]
            area = m["m00"]
            solidity = area / cv.contourArea(cv.convexHull(c))
        else:
            x = y = major = minor = area = solidity = 0
        rows.append([x / scale, y / scale, major / scale, minor / scale,
                     angle, area / scale**2, solidity])
    return np.array(rows, dtype=float).reshape(-1, 7)


def read_pixel_size(path):
    """Pixel size in µm from Zeiss LSM metadata, or None if not available."""
    try:
        with tifffile.TiffFile(path) as tif:
            if tif.is_lsm:
                return tif.lsm_metadata["VoxelSizeX"] * 1e6   # m -> µm
    except Exception:
        pass
    return None


def sort_nicely(items):
    """Sort strings the way humans expect (Image2 before Image10)."""
    def key(s):
        return [int(t) if t.isdigit() else t for t in re.split(r"([0-9]+)", s)]
    return sorted(items, key=key)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # --- Parameters (change to fit your experiment) ---
    data_dir = Path(__file__).resolve().parent  # folder with the images
    extension = ".lsm"                           # image file type
    crop = (slice(150, -40), slice(50, -100))    # rows, columns to keep
    scale_percent = 400                          # resize factor in %
    s_blur, lnoise, lobject, box_size = 1, 0.5, 7, 15
    min_aspect_ratio = 6
    min_area = 80                                # smallest object kept, in px (after resize)
    threshold_offset = 0                         # > 0 removes background noise, but also dim rods
    extra_formats = ["pdf", "svg"]                # also save the colored image in these formats
    show_figures = True                          # open figure windows (close them to end the script)
    polygon = [[0, 0], [0, 2000], [2000, 2000], [2000, 0]]  # region to keep

    fig_dir = data_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # --- Load the first image in the folder ---
    files = sort_nicely([f.name for f in data_dir.glob("*" + extension)])
    if not files:
        raise FileNotFoundError(f"No {extension} file found in {data_dir}")
    image_file = data_dir / files[0]
    print("Image:", image_file)
    # Hide OpenCV's warning about Zeiss LSM metadata (TIFF tag 34412)
    cv.utils.logging.setLogLevel(cv.utils.logging.LOG_LEVEL_ERROR)
    image = cv.imread(str(image_file), cv.IMREAD_UNCHANGED)
    if image is None:
        raise IOError(f"Could not read {image_file}")
    if image.ndim == 3:                          # color image -> grayscale
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    pixel_size = read_pixel_size(image_file)
    if pixel_size is None:
        pixel_size, unit = 1.0, "px"
        print("Pixel size not found: results in px of the original image")
    else:
        unit = "um"
        print(f"Pixel size: {pixel_size:.4f} um")

    # --- Crop and resize ---
    image = image[crop]
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    resized = cv.resize(image, (width, height), interpolation=cv.INTER_CUBIC)  # smooth upscaling
    print("Original size:", image.shape, " Resized:", resized.shape)

    # --- Filter and binarize ---
    filtered = unsharp(bpass(unsharp(mat2gray(resized, 255), s_blur, 0), lnoise, lobject),
                       lnoise, 0)
    bw = process_image(resized, s_blur=s_blur, s_amount=0, lnoise=lnoise,
                       lobject=lobject, box_size=box_size, small_blob_size=2 * min_area,
                       min_aspect_ratio=min_aspect_ratio,
                       threshold_offset=threshold_offset)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for ax, im, title in zip(axs, [resized, filtered, bw],
                             ["Original", "Filtered", "Binarized"]):
        ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(fig_dir / f"{image_file.stem}_steps.png", dpi=150, bbox_inches="tight")

    # --- Color by orientation ---
    colored, _ = color_image(bw, polygon)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(colored)
    ax.axis("off")
    fig.savefig(fig_dir / f"{image_file.stem}_colorcrop.png", dpi=150,
                bbox_inches="tight", pad_inches=0)
    # Vector version (rods as shapes, sharp at any zoom)
    if extra_formats:
        fig_v, ax_v = plt.subplots(figsize=(6, 6))
        plot_rods_vector(ax_v, bw, polygon)
        ax_v.set_xticks([])
        ax_v.set_yticks([])
        for fmt in extra_formats:
            fig_v.savefig(fig_dir / f"{image_file.stem}_colorcrop.{fmt}",
                          bbox_inches="tight", pad_inches=0)
        plt.close(fig_v)

    # --- Measure and export ---
    # Pixel size of the resized image: original pixel size / resize factor
    pixel_size_resized = pixel_size * 100 / scale_percent
    results = regionprops(bw, scale=1 / pixel_size_resized)
    out_file = data_dir / f"{image_file.stem}_X-Y-major-minor-Orient-Area-Solidity.dat"
    np.savetxt(out_file, results,
               header=(f"x({unit}) y({unit}) major({unit}) minor({unit}) "
                       f"orientation(deg) area({unit}^2) solidity"))
    print(f"{len(results)} objects saved to {out_file.name}")
    print(f"Median length: {np.median(results[:, 2]):.2f} {unit}, "
          f"median width: {np.median(results[:, 3]):.2f} {unit}")

    if show_figures:
        plt.show()
    plt.close("all")
    print("Done. Figures saved in", fig_dir)
