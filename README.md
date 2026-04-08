# 🌙 Lunar Surface Data Pipeline

Offline asset builder for a physically accurate WebGL2 lunar simulation.  
Downloads NASA **LRO** (albedo) and **LOLA** (elevation) GeoTIFFs, computes a tangent-space **normal map**, and compresses everything into GPU-ready **KTX2** textures.

---

## Dependencies

### 1. Python 3.9+

```bash
python3 --version   # must be ≥ 3.9
```

### 2. GDAL (with Python bindings)

GDAL is required to read/write GeoTIFF and PNG raster data.

**macOS (Homebrew):**
```bash
brew install gdal
pip install GDAL==$(gdal-config --version)
```

**Ubuntu / Debian:**
```bash
sudo apt install gdal-bin libgdal-dev python3-gdal
```

**Conda (cross-platform, easiest):**
```bash
conda install -c conda-forge gdal
```

### 3. NumPy

```bash
pip install numpy
```

### 4. SciPy *(optional — only used if the elevation GeoTIFF contains nodata pixels)*

```bash
pip install scipy
```

### 5. Basis Universal CLI (`basisu`)

Required **only** for the KTX2 compression step (skip with `--skip-ktx2`).

```bash
# Clone and build from source
git clone https://github.com/BinomialLLC/basis_universal.git
cd basis_universal
cmake -B build -DCMAKE_BUILD_TYPE=Release .
cmake --build build --config Release
# Copy the binary to somewhere on your PATH
sudo cp build/basisu /usr/local/bin/
```

Verify:
```bash
basisu --version
```

---

## Usage

### Full pipeline (download → process → KTX2)

```bash
python build_lunar_assets.py
```

### Custom output directory

```bash
python build_lunar_assets.py --output-dir ./my_assets
```

### Skip download (if you already have the raw TIFs)

Place your files as:
```
assets/lro_albedo_raw.tif
assets/lola_elevation_raw.tif
```

```bash
python build_lunar_assets.py --skip-download
```

### PNG-only (no basisu required)

```bash
python build_lunar_assets.py --skip-ktx2
```

---

## Output Files

| File | Description |
|---|---|
| `lunar_albedo.png` | 8-bit greyscale albedo map (normalised) |
| `lunar_normal.png` | RGB tangent-space normal map (OpenGL convention) |
| `lunar_albedo.ktx2` | UASTC-compressed albedo (GPU-ready) |
| `lunar_normal.ktx2` | UASTC-compressed normal map (linear, GPU-ready) |

---

## How the Normal Map Works

1. The LOLA elevation raster is read as `float64`.
2. The script calculates the **physical ground distance per pixel** using GDAL's GeoTransform and the IAU Moon radius (1,737.4 km).
3. A 3×3 **Sobel operator** computes ∂z/∂x and ∂z/∂y, scaled by the real pixel size in metres — this ensures crater slopes produce physically accurate shadows.
4. The tangent-space normal is constructed as **N = normalize(−∂z/∂x, −∂z/∂y, 1)** and encoded to RGB where `(128, 128, 255)` represents a flat surface pointing straight up.

---

## Swapping in Real NASA URLs

Edit the two placeholder constants at the top of `build_lunar_assets.py`:

```python
ALBEDO_URL = "https://..."    # 8K LRO WAC albedo GeoTIFF
ELEVATION_URL = "https://..."  # 8K LOLA DEM GeoTIFF
```

Recommended sources:
- **LRO WAC Global Morphology**: [LROC RDR Products (ASU)](https://wms.lroc.asu.edu/lroc/view_rdr_product/WAC_GLD100)
- **LOLA GDR Elevation**: [PDS Geosciences Node (WashU)](https://pds-geosciences.wustl.edu/missions/lro/lola.htm)
