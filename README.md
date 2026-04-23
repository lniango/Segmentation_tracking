# Cell Segmentation & Tracking

A comparative study of **image segmentation methods** applied to the temporal tracking of HeLa cells (cervical cancer cell line) in microscopy images. The dataset comes from the [Cell Tracking Challenge](https://celltrackingchallenge.net/2d-datasets/) (`DIC-C2DH-HeLa`), a standard benchmark in bio-imaging.

The goal is to **segment cells frame by frame**, then **track them over time** by following their evolution across the image sequence.

---

## Context

DIC (Differential Interference Contrast) images are grayscale microscopy images where cells appear with low contrast and poorly defined boundaries. This makes it a challenging segmentation task, and motivates the comparison of multiple approaches — from classical image processing to deep learning.

---

## Segmentation Methods

### 1. Binary Thresholding (`binary_seg`)
The simplest approach: each pixel is classified as foreground or background based on a fixed intensity threshold (120). Fast but not robust to contrast variations across images.

### 2. Canny Edge Detection (`canny_seg`)
Cell boundaries are detected using the Canny filter, followed by binary hole filling (`binary_fill_holes`) to recover full cell regions. Works well on clean edges but remains sensitive to noise.

### 3. Watershed Segmentation (`watershed_seg`)
A region-growing approach: a gradient map is first computed using a Sobel filter, then seed markers are placed based on **adaptive thresholds derived from the dataset's mean and standard deviation** (computed in `help.py`). The Watershed algorithm then floods from these markers to delineate cell regions.

```
low_threshold  = mean - k * std
high_threshold = mean + k * std
```

The parameter `k` controls sensitivity: a lower `k` gives tighter thresholds, a higher `k` is more permissive.

### 4. U-Net (Deep Learning)
A convolutional neural network trained **from scratch** on the HeLa dataset. U-Net is an encoder-decoder architecture with skip connections, originally designed for biomedical image segmentation. It predicts a binary mask for each image, separating cell pixels from the background.

Unlike classical methods, U-Net learns visual features directly from annotated data, making it significantly more robust to contrast variations and irregular cell shapes.

---

## Overall Pipeline

```
Input .tif images (temporal sequence)
        │
        ▼
Preprocessing (grayscale conversion, normalization)
        │
        ▼
Segmentation (Binary / Canny / Watershed / U-Net)
        │
        ▼
Binary masks per frame
        │
        ▼
Temporal tracking (cell association across frames)
```

---

## Project Structure

```
├── Seg_track.py          # Main script: classical segmentation methods
├── segmentation.ipynb    # Notebook: U-Net implementation and training
├── help.py               # Utilities: dataset mean and std computation
├── DIC-C2DH-HeLa/
│   ├── 01/               # Training image sequence (.tif)
│   └── 01_GT/            # Ground truth segmentation masks
└── output/
    ├── binary-seg/
    ├── canny_seg/
    └── watershed_seg/
```

---

## Installation

```bash
pip install opencv-python numpy pillow scikit-image scipy matplotlib torch torchvision
```

---

## Usage

```python
# Classical methods — in Seg_track.py
binary_seg(nb=10)          # Binary thresholding on the first 10 images
canny_seg(nb=10)           # Canny edge detection
watershed_seg(nb=10, k=1)  # Watershed segmentation (k=1 recommended)
```

For the U-Net, open and run `segmentation.ipynb`.

---

## References

- Dataset: [Cell Tracking Challenge – DIC-C2DH-HeLa](https://celltrackingchallenge.net/2d-datasets/)
- U-Net architecture: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- [Segment and Track Anything](https://github.com/z-x-yang/Segment-and-Track-Anything)
- [Seg2Track-SAM2](https://github.com/hcmr-lab/Seg2Track-SAM2)
