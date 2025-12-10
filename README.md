# Manual Image Processing Desktop App (Tkinter + NumPy)

Full desktop application that implements classic image processing and compression algorithms *manually* using NumPy (no cv2 resize/filter shortcuts). Aesthetic Tkinter GUI supports previews, saving, theme toggle, zoom, and side-by-side original/result views.

## Requirement Coverage (per coversheet)
- Image reading + info (resolution, size, type) via upload; dual canvases show original/current.
- Grayscale conversion using 0.299/0.587/0.114 weights.
- Binary conversion using average intensity threshold + optimality evaluation note.
- Affine transforms: translation, scaling, rotation, shear X, shear Y (manual matrices + bilinear).
- Interpolation (resolution enhancement): nearest neighbor, bilinear, bicubic (Catmull-Rom).
- Cropping: coordinate inputs or mouse drag on preview canvas.
- Histogram: 256-bin compute, qualitative assessment, manual histogram equalization with plotted view.
- Low-pass: 19×19 Gaussian (σ=3) convolution; 7×7 median filter.
- High-pass: Laplacian (second derivative), Sobel, first-derivative gradient.
- Compression (encode/decode, ratio, reconstructed preview): Huffman, Golomb–Rice, Arithmetic, LZW, Run-Length (RLE), Symbol-based, Bit-plane, Block DCT (8×8), Predictive (DPCM), Wavelet (Haar).
- GUI UX: grouped sections, dynamic preview-before-apply, save result, theme toggle (light/dark), zoom in/out, creative extras (preview/commit flow, dual canvases, histogram plot).

## Features
- Image load/save, info (resolution, size, type), original/current toggle via dual canvases.
- Grayscale (0.299/0.587/0.114) and binary with average threshold + quality note.
- Affine transforms: translation, scaling, rotation, shear X/Y (manual matrices + bilinear back-mapping).
- Interpolation/resolution enhancement: nearest, bilinear, bicubic (Catmull-Rom).
- Manual cropping via coordinates or drag selection on preview canvas.
- Histogram: 256-bin compute, qualitative evaluation, custom drawn plot, manual equalization.
- Low-pass: 19×19 Gaussian (σ=3) convolution, 7×7 median.
- High-pass: Laplacian, Sobel, first-derivative gradient.
- Compression (encode/decode + ratio + PSNR preview):
  - Huffman, Golomb–Rice, Arithmetic, LZW, RLE, Symbol-based, Bit-plane.
  - Block DCT 8×8 with quantization.
  - Predictive (DPCM).
  - Wavelet (Haar, thresholding).
- Extra UX: preview before apply, Apply Preview button, theme toggle, zoom in/out.

## Project Structure
```
src/
  main.py
  gui.py
  utils.py
  image_loader.py
  grayscale.py
  binary.py
  affine.py
  interpolation.py
  histogram.py
  filters.py
  highpass.py
  compression/
      __init__.py
      huffman.py
      golomb_rice.py
      arithmetic.py
      lzw.py
      rle.py
      bitplane.py
      dct.py
      predictive.py
      wavelet.py
      symbol_based.py
```

## How to Run
1. Install requirements (Pillow, NumPy; Tk is bundled with most Python installs):
   ```bash
   pip install numpy pillow
   ```
2. Launch GUI:
   ```bash
   python -m src.main
   ```
   or `python src/main.py` if the current directory is `src`.

## Using the App
- **Load**: Click *Upload Image* and select a color image; info appears in the toolbar.
- **Preview then Apply**: Any operation renders to the preview pane; click *Apply Preview* to commit as the current image. *Reset* restores the original.
- **Cropping**: Enter coordinates and *Crop by coords*, or drag on the preview canvas; the selection is mapped back to the original pixel grid.
- **Histogram**: *Compute Histogram* shows plot/evaluation; *Equalize* applies manual histogram equalization.
- **Affine / Interpolation**: Fill parameters and click the desired transform or resize method.
- **Filters / Edges**: Buttons apply Gaussian/median or Laplacian/Sobel/gradient.
- **Compression**: Each algorithm encodes/decodes, then shows compression ratio (original bits ÷ compressed bits) and PSNR; the reconstructed image is previewed.
- **Theme & Zoom**: *Theme Toggle* switches light/dark; *Zoom +/-* scales the display while keeping processing at native resolution.
- **Save**: *Save Result* writes the current image to disk.

## Notes on Manual Implementations
- Convolutions, interpolations, and transforms iterate pixel grids directly (NumPy loops/vector math) without OpenCV shortcuts.
- Bilinear sampling and Catmull-Rom bicubic are implemented from scratch.
- Compression algorithms store simple serializations (bitstrings or arrays) to demonstrate concepts and compute ratios.
- DCT uses explicit cosine sums with an 8×8 quantization matrix; Wavelet uses one-level Haar with coefficient thresholding.

## Troubleshooting
- If Tk windows fail to open on some platforms, ensure a GUI backend is available.
- Large images with big kernels (19×19 Gaussian) can be slow; consider testing with medium resolutions first.

