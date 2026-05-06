# Cuticle Melanization Analyzer

A desktop GUI tool for quantifying cuticle (skin) melanization in caterpillars from overhead images. Designed for researchers and students, no programming experience required to use.

---

## What It Does

- Detects and outlines the caterpillar body from images taken on a white background
- Generates a false-color heatmap showing melanization intensity across the body
- Calculates **whole-body mean pixel intensity**
- Traces a dorsal centerline and calculates **mean pixel intensity along the centerline**
- Plots the intensity profile from head to tail
- Exports results as CSV files and a heatmap PNG
- Batch-processes entire folders and generates an **average heatmap** across individuals

---

## Requirements

- **Python 3.9 or newer** — [Download Python](https://www.python.org/downloads/)
- The following Python packages (installed in one command below):

| Package | Purpose |
|---|---|
| `numpy` | Numerical array operations |
| `opencv-python` | Image loading, segmentation, drawing |
| `scikit-image` | Centerline skeletonization |
| `scipy` | Spline smoothing of the centerline |
| `matplotlib` | Heatmap rendering and embedded plots |
| `pandas` | CSV export |

---

## Installation

### 1. Make sure Python is installed

Open a terminal (Mac/Linux) or Command Prompt (Windows) and run:

```bash
python3 --version
```

You should see something like `Python 3.11.x`. If not, install Python from https://www.python.org/downloads/

### 2. Install dependencies

Navigate to the folder containing the project files, then run:

```bash
pip install -r requirements.txt 
```

> **Windows users:** use `pip` instead of `pip3` if the above doesn't work.  
> **Mac users:** if you get a permissions error, add `--user` to the command.

### 3. Run the app

```bash
python3 melanization_analyzer.py
```

> **Windows users:** you may be able to double-click `melanization_analyzer.py` to launch it directly if Python is associated with `.py` files.

---

## Image Requirements

- Caterpillar photographed **from above** on a **plain white or light background**
- Any camera works (Nikon, phone, etc.)
- Supported formats: **JPG, JPEG, PNG, TIFF, TIF, BMP**
- One caterpillar per image recommended for best results

---

## How to Use

### Single Image Analysis

1. Click **Open Image…** (or File → Open Image…) and select your photo
2. Click **Process Image**
   - The app will auto-detect the caterpillar using Otsu thresholding
   - Results appear in the **Results** panel on the right
3. Use the **View** buttons at the top to switch between:
   - **Original** — your raw photo
   - **Grayscale** — converted grayscale
   - **Heatmap** — false-color map of melanization intensity
   - **Heatmap + Overlay** — heatmap with green body contour and yellow dorsal centerline drawn on top
4. Click **View Intensity Profile…** to open a plot of intensity along the centerline from one end of the body to the other
5. Click **Export Results…** and choose an output folder

### Exported Files (per image)

| File | Contents |
|---|---|
| `<name>_summary.csv` | Whole-body mean intensity and dorsal centerline mean intensity |
| `<name>_centerline_profile.csv` | Per-point intensity values along the centerline |
| `<name>_heatmap.png` | Heatmap image with contour and centerline overlay |

### Batch Processing (Multiple Images)

1. Click **Open Folder…** or **Select Folder…** in the Batch Processing panel
2. Click **Run Batch**
3. Choose an output folder
4. The app processes all images in the folder and saves:
   - Individual result files for each image (same as single-image export)
   - `batch_summary.csv` — one row per image with both intensity measurements
   - `average_heatmap.png` — pixel-averaged heatmap across all individuals

---

## Controls Reference

| Control | Description |
|---|---|
| **Threshold slider** | `0` = automatic (Otsu method, recommended). Move right to manually raise the cutoff if the auto-detection picks up shadows or background debris. |
| **Colormap dropdown** | Color scheme for the heatmap. `inferno` and `plasma` are recommended for melanization work. |
| **CL Smooth slider** | Controls how much the centerline is smoothed. `1.0` works for most specimens; increase for curled or irregular caterpillars. |

---

## Interpreting Results

**Pixel intensity** is measured on a 0–255 scale:

- **Lower value = darker pixel = more melanized**
- **Higher value = lighter pixel = less melanized**

On the heatmap, the color scale is inverted to be intuitive:

- **Brighter / hotter colors** (yellow, white) = darker in the original photo = **more melanized**
- **Cooler / darker colors** (purple, black) = lighter in the original photo = **less melanized**

---

## Troubleshooting

**The contour includes background or misses parts of the body**  
→ Adjust the **Threshold** slider. Move it right (higher value) to exclude more of the background; move it left if the body is being cut off.

**The centerline looks jagged or goes off the body**  
→ Increase the **CL Smooth** slider. Also check that the mask looks correct first (view in Heatmap + Overlay).

**The app crashes on a specific image**  
→ Check that the image is not corrupted. Very large RAW files (e.g., `.NEF`, `.CR2`) are not supported — export from your camera software as TIFF or JPEG first.

**`pip install` fails**  
→ Try `pip3 install -r requirements.txt`. On Windows, make sure Python was added to PATH during installation.

**Nothing appears after clicking Process Image**  
→ The caterpillar may not be distinct enough from the background. Try a photo with better contrast or manually adjust the threshold slider.

---

## File Overview

```
Cuticle_Melanization/
├── melanization_analyzer.py   # GUI application — run this to launch
├── image_processing.py        # Image analysis backend (no GUI)
├── requirements.txt           # Python package dependencies
└── README.md                  # This file
```

---

## Citation / Acknowledgments

If you use this tool in published research, please acknowledge it in your methods section.
