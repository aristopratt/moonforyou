#!/usr/bin/env python3
"""
Lunar Surface Data Processing Pipeline
=======================================
Downloads NASA LRO (Albedo) and LOLA (Elevation) GeoTIFFs, generates a
tangent-space normal map from the elevation data, exports PNG intermediates,
and compresses them into GPU-ready KTX2 (UASTC) via the basisu CLI.

Usage:
    python build_lunar_assets.py [--output-dir ./assets] [--skip-download] [--skip-ktx2]

Dependencies (see README.md):
    - Python 3.9+
    - GDAL (with Python bindings)
    - NumPy
    - basisu CLI (for KTX2 compression step)
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import urllib.request
import ssl
from pathlib import Path

# Bypass SSL verify on Mac
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np

try:
    from osgeo import gdal
except ImportError:
    import gdal  # type: ignore

# ---------------------------------------------------------------------------
# Source URLs  (PLACEHOLDER – swap with real NASA PDS / LROC node URLs)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Source URLs (NASA SVS CGI Moon Kit)
# ---------------------------------------------------------------------------
ALBEDO_URL = (
    "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/lroc_color_16bit_srgb_8k.tif"
    # ^^^ placeholder – replace with the actual 8K LRO WAC albedo GeoTIFF URL
)

ELEVATION_URL = (
 "https://svs.gsfc.nasa.gov/vis/a000000/a004700/a004720/ldem_16.tif"
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = Path("assets")
ALBEDO_RAW = "lro_albedo_raw.tif"
ELEVATION_RAW = "lola_elevation_raw.tif"
ALBEDO_PNG = "lunar_albedo.png"
NORMAL_PNG = "lunar_normal.png"
ALBEDO_KTX2 = "lunar_albedo.ktx2"
NORMAL_KTX2 = "lunar_normal.ktx2"

# Moon equatorial radius in metres (IAU 2015)
MOON_RADIUS_M = 1_737_400.0


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  1. Downloading                                                       ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def download_file(url: str, dest: Path) -> Path:
    """Download *url* to *dest*, showing a simple progress indicator."""
    if dest.exists():
        print(f"  ✓ Already cached: {dest}")
        return dest

    print(f"  ↓ Downloading: {url}")
    print(f"    → {dest}")

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100.0, downloaded / total_size * 100)
            bar = "█" * int(pct // 2) + "░" * (50 - int(pct // 2))
            print(f"\r    [{bar}] {pct:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=_report)
    print()  # newline after progress bar
    return dest


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  2. GeoTIFF → 16-bit / 8-bit PNG Export                              ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def geotiff_to_8bit_png(src_path: Path, dst_path: Path) -> None:
    """
    Read a single-band GeoTIFF, normalise to 0-255, and write an 8-bit PNG.
    Suitable for the albedo channel.
    """
    print(f"  → Converting {src_path.name} → {dst_path.name}")
    ds: gdal.Dataset = gdal.Open(str(src_path), gdal.GA_ReadOnly)
    if ds is None:
        sys.exit(f"ERROR: Cannot open {src_path}")

    band: gdal.Band = ds.GetRasterBand(1)
    arr: np.ndarray = band.ReadAsArray().astype(np.float64)

    # Normalise to [0, 255]
    lo, hi = np.nanmin(arr), np.nanmax(arr)
    if hi - lo == 0:
        sys.exit("ERROR: Albedo raster has zero dynamic range.")
    arr = (arr - lo) / (hi - lo) * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    height, width = arr.shape
    mem_driver: gdal.Driver = gdal.GetDriverByName("MEM")
    mem_ds: gdal.Dataset = mem_driver.Create("", width, height, 1, gdal.GDT_Byte)
    mem_ds.GetRasterBand(1).WriteArray(arr)
    
    png_driver: gdal.Driver = gdal.GetDriverByName("PNG")
    png_driver.CreateCopy(str(dst_path), mem_ds)
    
    mem_ds = None
    ds = None
    print(f"  ✓ Wrote {dst_path}  ({width}×{height} px)")


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  3. Elevation → Tangent-Space Normal Map                              ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def _compute_pixel_size_metres(ds: gdal.Dataset) -> tuple[float, float]:
    """
    Return (dx_m, dy_m) — the ground distance in metres that one pixel spans
    in the X and Y directions.

    For geographic (lon/lat) projections the result depends on latitude; we
    use the equator as reference (worst-case / maximum spacing).  For
    projected CRSs GDAL already gives us metres directly.
    """
    gt = ds.GetGeoTransform()
    pixel_x_deg = abs(gt[1])
    pixel_y_deg = abs(gt[5])

    proj = ds.GetProjection()
    # Simple heuristic: if the unit looks like degrees, convert via Moon radius
    if "GEOGCS" in proj or pixel_x_deg < 1.0:
        # degrees → radians → arc length at equator
        dx_m = np.radians(pixel_x_deg) * MOON_RADIUS_M
        dy_m = np.radians(pixel_y_deg) * MOON_RADIUS_M
    else:
        # Already in a projected CRS (metres)
        dx_m = pixel_x_deg
        dy_m = pixel_y_deg

    return dx_m, dy_m


def generate_normal_map(elevation_path: Path, normal_png_path: Path) -> None:
    """
    Generate a tangent-space normal map from LOLA elevation data.

    The function:
      1. Reads the DEM as float64.
      2. Computes per-pixel slopes (∂z/∂x, ∂z/∂y) using a 3×3 Sobel operator
         scaled by the *physical* pixel size in metres so that crater shadows
         are geometrically accurate.
      3. Constructs the tangent-space normal  N = normalise(-dz/dx, -dz/dy, 1).
      4. Encodes into an 8-bit RGB PNG  (R = Nx, G = Ny, B = Nz) mapped from
         [-1, 1] → [0, 255].
    """
    print(f"  → Generating normal map from {elevation_path.name}")

    ds: gdal.Dataset = gdal.Open(str(elevation_path), gdal.GA_ReadOnly)
    if ds is None:
        sys.exit(f"ERROR: Cannot open {elevation_path}")

    band: gdal.Band = ds.GetRasterBand(1)
    elev: np.ndarray = band.ReadAsArray().astype(np.float64)

    # Replace nodata with local mean to avoid edge artefacts
    nodata = band.GetNoDataValue()
    if nodata is not None:
        mask = elev == nodata
        if mask.any():
            from scipy.ndimage import uniform_filter
            elev[mask] = uniform_filter(np.where(mask, 0, elev), size=5)[mask]

    dx_m, dy_m = _compute_pixel_size_metres(ds)
    print(f"    Pixel ground size: {dx_m:.2f} m × {dy_m:.2f} m")

    height, width = elev.shape

    # --- Sobel-based partial derivatives --------------------------------
    # Sobel kernels weight the centre row/column by 2, giving a smoothed
    # finite-difference over a 3×3 neighbourhood.
    #
    #   dz/dx ≈  [(-1, 0, +1),    /  (8 · dx)
    #             (-2, 0, +2),
    #             (-1, 0, +1)]
    #
    # Numpy implementation avoids explicit convolution for speed on 8K grids.

    # Pad elevation by 1 pixel on each side (reflect boundary)
    elev_pad = np.pad(elev, 1, mode="reflect")

    # Horizontal gradient  (∂z/∂x)
    dzdx = (
        -1.0 * elev_pad[0:-2, 0:-2]
        + 1.0 * elev_pad[0:-2, 2:]
        - 2.0 * elev_pad[1:-1, 0:-2]
        + 2.0 * elev_pad[1:-1, 2:]
        - 1.0 * elev_pad[2:, 0:-2]
        + 1.0 * elev_pad[2:, 2:]
    ) / (8.0 * dx_m)

    # Vertical gradient  (∂z/∂y)  — note: image Y is *south*, so negate for
    # a right-handed tangent space that matches OpenGL / WebGL conventions.
    dzdy = (
        -1.0 * elev_pad[0:-2, 0:-2]
        - 2.0 * elev_pad[0:-2, 1:-1]
        - 1.0 * elev_pad[0:-2, 2:]
        + 1.0 * elev_pad[2:, 0:-2]
        + 2.0 * elev_pad[2:, 1:-1]
        + 1.0 * elev_pad[2:, 2:]
    ) / (8.0 * dy_m)

    # --- Build tangent-space normals ------------------------------------
    # N = normalise(-dz/dx, -dz/dy, 1)
    nx = -dzdx
    ny = -dzdy  # already accounts for image-Y flip via the kernel sign
    nz = np.ones_like(nx)

    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    length[length == 0] = 1.0  # guard
    nx /= length
    ny /= length
    nz /= length

    # --- Encode to 8-bit RGB  [-1,1] → [0,255] -------------------------
    def _encode(channel: np.ndarray) -> np.ndarray:
        return np.clip((channel * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)

    r = _encode(nx)
    g = _encode(ny)
    b = _encode(nz)

    # Write 3-band PNG via GDAL CreateCopy
    mem_driver: gdal.Driver = gdal.GetDriverByName("MEM")
    mem_ds: gdal.Dataset = mem_driver.Create("", width, height, 3, gdal.GDT_Byte)
    mem_ds.GetRasterBand(1).WriteArray(r)
    mem_ds.GetRasterBand(2).WriteArray(g)
    mem_ds.GetRasterBand(3).WriteArray(b)
    
    png_driver: gdal.Driver = gdal.GetDriverByName("PNG")
    png_driver.CreateCopy(str(normal_png_path), mem_ds)
    
    mem_ds = None
    ds = None

    print(f"  ✓ Wrote {normal_png_path}  ({width}×{height} px, RGB tangent-space)")


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  4. PNG → KTX2 (UASTC via basisu CLI)                                ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def compress_to_ktx2(png_path: Path, ktx2_path: Path, is_normal: bool = False) -> None:
    """
    Shell out to the **basisu** CLI encoder to produce a UASTC-compressed
    KTX2 file.  For normal maps we pass ``-normal_map`` so the encoder
    treats the data in linear space and avoids chroma sub-sampling.
    """
    print(f"  → Compressing {png_path.name} → {ktx2_path.name}  (UASTC)")

    cmd = [
        "basisu",
        str(png_path),
        "-ktx2",               # output KTX2 container
        "-uastc",              # use UASTC super-compression
        "-uastc_level", "3",   # quality level (0=fastest, 4=best)
        "-uastc_rdo_l", "1.0", # rate-distortion optimisation lambda
        "-mipmap",             # generate a full mip chain
        "-output_file", str(ktx2_path),
    ]

    if is_normal:
        cmd += [
            "-normal_map",     # linear-space, no chroma tricks
        ]

    print(f"    $ {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            # Print last few informational lines
            for line in result.stdout.strip().splitlines()[-5:]:
                print(f"      {line}")
        print(f"  ✓ Wrote {ktx2_path}")
    except FileNotFoundError:
        print(
            "  ✗ ERROR: 'basisu' not found on PATH.\n"
            "    Install from https://github.com/BinomialLLC/basis_universal\n"
            "    or set --skip-ktx2 to produce PNGs only."
        )
        sys.exit(1)
    except subprocess.CalledProcessError as exc:
        print(f"  ✗ basisu failed (exit {exc.returncode}):\n{exc.stderr}")
        sys.exit(1)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  Main Pipeline                                                        ║
# ╚═════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lunar Surface Asset Builder — LRO albedo + LOLA normals → KTX2"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory for all output files (default: ./assets)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip downloading; assume raw TIFs already exist in output dir.",
    )
    parser.add_argument(
        "--skip-ktx2", action="store_true",
        help="Skip the basisu KTX2 compression step (output PNGs only).",
    )
    args = parser.parse_args()

    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    albedo_tif = out / ALBEDO_RAW
    elev_tif = out / ELEVATION_RAW
    albedo_png = out / ALBEDO_PNG
    normal_png = out / NORMAL_PNG
    albedo_ktx2 = out / ALBEDO_KTX2
    normal_ktx2 = out / NORMAL_KTX2

    # ── Step 1: Download ──────────────────────────────────────────────────
    if not args.skip_download:
        print("\n═══ Step 1/4: Downloading source GeoTIFFs ═══")
        download_file(ALBEDO_URL, albedo_tif)
        download_file(ELEVATION_URL, elev_tif)
    else:
        print("\n═══ Step 1/4: Download skipped (--skip-download) ═══")
        if not albedo_tif.exists():
            sys.exit(f"ERROR: Expected {albedo_tif} but file not found.")
        if not elev_tif.exists():
            sys.exit(f"ERROR: Expected {elev_tif} but file not found.")

    # ── Step 2: Albedo → PNG ──────────────────────────────────────────────
    print("\n═══ Step 2/4: Albedo GeoTIFF → PNG ═══")
    geotiff_to_8bit_png(albedo_tif, albedo_png)

    # ── Step 3: Elevation → Normal Map PNG ────────────────────────────────
    print("\n═══ Step 3/4: LOLA Elevation → Tangent-Space Normal Map ═══")
    generate_normal_map(elev_tif, normal_png)

    # ── Step 4: KTX2 compression ──────────────────────────────────────────
    if not args.skip_ktx2:
        print("\n═══ Step 4/4: PNG → KTX2 (UASTC via basisu) ═══")
        compress_to_ktx2(albedo_png, albedo_ktx2, is_normal=False)
        compress_to_ktx2(normal_png, normal_ktx2, is_normal=True)
    else:
        print("\n═══ Step 4/4: KTX2 compression skipped (--skip-ktx2) ═══")

    # ── Done ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Pipeline complete!  Output files:")
    for f in [albedo_png, normal_png, albedo_ktx2, normal_ktx2]:
        tag = "  ✓" if f.exists() else "  ·"
        print(f"  {tag} {f}")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
