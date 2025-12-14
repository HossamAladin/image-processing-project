# Quick Start Guide

## Project Status: ✅ FULLY COMPLIANT

Your project meets all requirements from the Image Processing Project PDF. See `REQUIREMENTS_COMPLIANCE.md` for detailed verification.

## Running the Application

### Method 1: From project root
```bash
py -m src.main
```

### Method 2: From src directory
```bash
cd src
py main.py
```

## How to Use

1. **Upload Image**: Click "Upload Image" button and select a colored image (PNG, JPG, JPEG, BMP)

2. **View Image Info**: After uploading, the toolbar shows:
   - Resolution (width x height)
   - File size in bytes
   - Image type (RGB/Grayscale)

3. **Apply Operations**: 
   - Click any operation button to see a preview
   - Click "Apply Preview" to commit the change
   - Click "Reset" to restore the original image

4. **Available Operations**:
   - **Basic**: Grayscale, Binary conversion
   - **Affine**: Translation, Scaling, Rotation, Shear X/Y
   - **Interpolation**: Nearest, Bilinear, Bicubic resizing
   - **Histogram**: Compute histogram (with plot), Equalize
   - **Low-Pass Filters**: Gaussian 19×19 (σ=3), Median 7×7
   - **High-Pass Filters**: Laplacian, Sobel, Gradient
   - **Compression**: All 10 compression algorithms with ratio/PSNR display
   - **Crop**: Enter coordinates or drag on preview canvas

5. **Extra Features**:
   - **Theme Toggle**: Switch between light/dark mode
   - **Zoom +/-**: Zoom in/out on displayed images
   - **Save Result**: Save the current processed image

## Testing All Features

To test all features systematically:

1. **Image Reading**: Upload an image → Check info display
2. **Grayscale**: Click "Grayscale" → Apply Preview
3. **Binary**: Click "Binary" → Check threshold evaluation message
4. **Affine**: Test each transform with different parameters
5. **Interpolation**: Resize using all three methods
6. **Crop**: Try both coordinate input and mouse drag
7. **Histogram**: Click "Compute Histogram" → View plot and evaluation → Try "Equalize"
8. **Filters**: Test Gaussian and Median filters
9. **High-Pass**: Test Laplacian, Sobel, and Gradient
10. **Compression**: Test each compression algorithm (note: some may be slow on large images)

## Notes

- All operations use **custom implementations** (no OpenCV shortcuts)
- Preview workflow: Operations show in preview pane first, then apply when ready
- Large images with big kernels (19×19 Gaussian) may take a few seconds
- Compression algorithms show compression ratio and PSNR quality metrics

## Troubleshooting

- If GUI doesn't open: Ensure Python Tkinter is installed (usually bundled)
- If operations are slow: Try with smaller images first
- If import errors: Make sure you're running from the correct directory

