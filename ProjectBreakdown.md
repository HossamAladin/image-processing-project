## Image Processing Project – Conceptual & Technical Breakdown

This document explains every method, transform, filter, and compression technique used in the Image Processing project. It is written as a study guide for exams or viva, focusing on **concepts**, **motivations**, and **implementation logic**, rather than raw code.

---

## 1. Introduction

### 1.1 What the Project Does

This project is a **desktop image-processing application** built in **Python + Tkinter + NumPy**. It:

- Loads a color image from disk.
- Displays image information (resolution, size, type).
- Applies a wide range of **spatial-domain** operations:
  - Grayscale & binary conversion
  - Affine transformations (translation, scaling, rotation, shear)
  - Interpolation-based resizing (nearest, bilinear, bicubic)
  - Cropping (by coordinates and mouse selection)
  - Histogram analysis & equalization
  - Low-pass and high-pass filtering
- Implements many **compression techniques**:
  - Huffman, Golomb–Rice, Arithmetic, LZW, RLE, Symbol-based
  - Bit-plane, DCT, Predictive (DPCM), Wavelet (Haar)
- Shows **compression ratio**, **reconstructed image**, and **quality (PSNR)**.

All major algorithms are implemented **manually with NumPy**, without relying on OpenCV/PIL functions that hide the logic.

### 1.2 High-Level Workflow

1. **User loads an image** via the GUI.
2. The image is converted to a NumPy array and stored as the **original** and **current** images.
3. The user selects an operation (e.g., Grayscale, Rotate, Huffman).
4. The corresponding algorithm runs on the **current image** and generates a **preview result**.
5. The preview is shown side-by-side with the original.
6. If satisfied, the user clicks **“Apply Preview”** to make the preview the new current image.
7. At any time, the user can:
   - **Reset** back to the original.
   - **Save** the current result.
   - Zoom in/out or toggle theme (light/dark).

### 1.3 Why These Methods Are Grouped Together

The project is designed to cover the main pillars of an introductory image-processing course:

- **Basic Intensity Operations**  
  Grayscale, binary, histogram, equalization.

- **Geometric Transformations**  
  Affine transforms and interpolation answer: *“What happens when the coordinates or size change?”*

- **Spatial Filtering**  
  Low-pass and high-pass filters answer: *“How do we smooth or enhance details/edges?”*

- **Compression Techniques**  
  Multiple coding schemes answer: *“How can we reduce size while preserving acceptable visual quality?”*

These groups together show a **full pipeline**:

> Real-world image → preprocessing → analysis / enhancement → compression → storage/transmission

### 1.4 Math & Notation Conventions

To keep the rest of this document easy to read, we use the following consistent notation:

- \(x, y\): spatial coordinates (column and row indices) in the image grid.
- \(x', y'\): transformed coordinates after a geometric operation.
- \(f(x,y)\): original image intensity at \((x,y)\) (grayscale).
- \(R, G, B\): red, green, blue channel values in an RGB image.
- \(Y\): grayscale (luminance) intensity.
- \(T\): threshold value for binary conversion.
- \(G(x,y)\): value of a **Gaussian kernel** at position \((x,y)\).
- \(C(u,v)\): DCT coefficient at frequency indices \((u,v)\).
- Bold symbols like \(\mathbf{A}\): matrices (e.g., affine transform matrix).

Whenever we show formulas, we write them in standard mathematical form, for example:

\[
Y(x,y) = 0.299\,R(x,y) + 0.587\,G(x,y) + 0.114\,B(x,y)
\]

This makes it easier to connect the **equations** to the **code logic** that implements them.

---

## 2. Detailed Breakdown of Every Methodology

For each method below, we cover:

- **A. What it does**
- **B. Why it is used**
- **C. When to apply it**
- **D. Logic behind the method**
- **E. What happens if you do NOT use it**
- **F. Expected results when applied correctly**
- **G. Simple intuition + mathematical intuition**

---

### 2.1 Image Reading

**A. What it does**  
Reads an image file (e.g., PNG/JPEG), converts it to a **NumPy array**, and extracts:
- Width × Height  
- File size in bytes  
- Type (RGB or Grayscale)

**B. Why it is used**  
All operations work on pixel arrays. We must first map the disk file → in-memory **matrix of pixels**.

**C. When to apply it**  
Always the first step before any image processing.

**D. Logic behind the method**
- Use `PIL.Image.open(path).convert("RGB")` for consistent 3-channel images.
- Convert to NumPy: shape `(H, W, 3)`.
- Use `os.path.getsize(path)` for size.

ASCII view:

```text
File on disk ──> PIL Image ──> NumPy array (H × W × 3)
```

**E. If you don’t use it**  
You have no pixel data → no processing → the project cannot function.

**F. Expected results**  
- Correct image display.
- Correct metadata: `width`, `height`, `size_bytes`, `mode`.

**G. Intuition**  
Conceptually, an image is just a **2D or 3D matrix of numbers**. Image reading is the “decoder” from disk format to this matrix.

---

### 2.2 Grayscale Conversion

**A. What it does**  
Converts RGB color image to single-channel grayscale using:
\[
Y = 0.299 R + 0.587 G + 0.114 B
\]

**B. Why it is used**  
Many algorithms (histogram, edge detection, some compression) operate on **intensity**, not color. Grayscale simplifies computation and theory.

**C. When to apply it**
- Before **binary thresholding**.
- Before **edge detection** (Sobel, Laplacian).
- Before **many compression experiments** (for conceptual clarity).

**D. Logic behind the method**
- Human vision is more sensitive to **green**, less to **blue**.
- The weights 0.299/0.587/0.114 approximate human luminance perception.
- Implementation loops over pixels or uses vectorized operations:

```python
gray = 0.299 * R + 0.587 * G + 0.114 * B
```

**E. If you don’t use it**  
Binary thresholding on individual RGB channels becomes confusing. Edge detection in color can be noisy and hard to interpret. The theory is also more complex.

**F. Expected results**  
Clear grayscale version maintaining brightness structure of the original scene.

**G. Intuition**  
- **Simple**: Turn color photo into black-and-white while preserving brightness relationships.  
- **Math**: Weighted sum of three channels → projection of 3D color vector onto a 1D luminance axis.

---

### 2.3 Binary Conversion + Threshold Evaluation

**A. What it does**  
Converts grayscale to a **binary (black & white)** image via a threshold \( T \):
\[
B(x,y) =
\begin{cases}
255, & \text{if } G(x,y) \ge T \\
0,   & \text{otherwise}
\end{cases}
\]

Here, \( T \) is the **average intensity** of all pixels.

**B. Why it is used**  
Binary images are used for **segmentation**, **object/background separation**, or **masking**.

**C. When to apply it**
- When you only care about **foreground vs background**, not exact gray levels.
- For simple document images, shapes, or high-contrast objects.

**D. Logic behind the method**
1. Compute average:
   \[
   T = \frac{1}{N} \sum_{x,y} G(x,y)
   \]
2. Compare each pixel to \( T \).
3. Evaluate threshold quality by checking fraction of pixels below/above \( T \). If one side dominates heavily, threshold might be poor.

**E. If you don’t use it**  
You keep full grayscale; segmentation decisions become more complex. You cannot easily classify “on/off” regions.

**F. Expected results**  
- Foreground objects (brighter than average) become **white**.
- Background (darker than average) becomes **black**.
- Evaluation text in GUI tells if threshold is **balanced or biased**.

**G. Intuition**
- **Simple**: Split the image into “bright enough” vs “too dark” according to the global average.  
- **Math**: Global mean is a crude estimator of an optimal threshold; more advanced methods (Otsu) minimize intra-class variance, but average is a good pedagogical start.

---

### 2.4 Affine Transformations

Affine transforms are **linear transformations + translation** in homogeneous coordinates:
\[
\begin{bmatrix}
x' \\ y' \\ 1
\end{bmatrix}

=

\mathbf{A}
\begin{bmatrix}
x \\ y \\ 1
\end{bmatrix}
\]

The implementation uses **backward mapping** + **bilinear interpolation**.

#### 2.4.1 Translation

**A. What it does**  
Shifts the image by \((T_x, T_y)\) pixels.

**B. Why it is used**  
To move objects around, align images, or create animations.

**C. When to apply it**
- Registration problems, or to center a region of interest.

**D. Logic**  
Matrix:
\[
A =
\begin{bmatrix}
1 & 0 & T_x\\
0 & 1 & T_y\\
0 & 0 & 1
\end{bmatrix}
\]
For each output pixel \((x', y')\) we compute source coordinate:
\[
[x, y, 1]^T = A^{-1} [x', y', 1]^T
\]
Then sample source via bilinear interpolation.

**E. If not used**  
You cannot reposition images or align them.

**F. Expected results**  
Same image content, simply shifted, with blank regions near edges.

**G. Intuition**  
Just adding offsets to x and y coordinates.

#### 2.4.2 Scaling

**A. What it does**  
Enlarges or shrinks the image by factors \(S_x, S_y\).

**B. Why**  
To change resolution, zoom level, or fit an image into a specific UI layout.

**C. When**  
Resizing images for display, matching sizes for further operations.

**D. Logic**
\[
A =
\begin{bmatrix}
S_x & 0 & 0\\
0 & S_y & 0\\
0 & 0 & 1
\end{bmatrix}
\]
Output size is scaled; backward mapping uses \(A^{-1}\).

**E. Without it**  
You cannot change image size properly; naive resampling causes strong artifacts.

**F. Expected results**  
Properly scaled image with smooth interpolation (thanks to bilinear sampling).

**G. Intuition**  
Multiplying coordinates by a factor stretches or compresses the grid.

#### 2.4.3 Rotation

**A. What it does**  
Rotates image around the origin (or around the image center if adjusted) by angle \(\theta\).

**B. Why**  
Align tilted images, perform augmentations, or rotate camera views.

**C. When**  
Rotating scanned documents, aligning views, augmenting datasets.

**D. Logic**
\[
A =
\begin{bmatrix}
\cos\theta & -\sin\theta & 0\\
\sin\theta & \cos\theta & 0\\
0 & 0 & 1
\end{bmatrix}
\]
Backward mapping + bilinear interpolation again.

**E. Without it**  
You cannot correct orientation; text/images might remain tilted.

**F. Expected results**  
Image appears rotated; some corners may be cut off (depending on output frame).

**G. Intuition**
- Rotating points on a plane using standard 2D rotation formulas.  
- Geometrically: we spin the coordinate grid under the image.

#### 2.4.4 Shear (X & Y)

**A. What it does**  
Applies a “slant” in X or Y direction.

**B. Why**  
To simulate perspective-like effects or perform certain geometric corrections.

**C. When**  
Data augmentation; geometry experiments.

**D. Logic**
- X-shear:
  \[
  A =
  \begin{bmatrix}
  1 & s_x & 0\\
  0 & 1 & 0\\
  0 & 0 & 1
  \end{bmatrix}
  \]
- Y-shear:
  \[
  A =
  \begin{bmatrix}
  1 & 0 & 0\\
  s_y & 1 & 0\\
  0 & 0 & 1
  \end{bmatrix}
  \]

**E. Without it**  
Limited ability to simulate perspective or to perform some geometric warps.

**F. Expected results**  
Image appears slanted like a parallelogram.

**G. Intuition**  
Shear adds a multiple of one coordinate into the other: \(x' = x + s_x y\) (for X-shear), etc.

---

### 2.5 Image Interpolation Methods

Interpolation is about estimating **pixel values at non-integer coordinates**, primarily for resizing.

#### 2.5.1 Nearest Neighbor

**A. What it does**  
Assigns each output pixel the value of the **nearest** input pixel.

**B. Why**  
Fastest, simplest; keeps sharp edges for certain graphics or label maps.

**C. When**  
When speed matters more than visual smoothness; when dealing with masks/labels.

**D. Logic**
- Compute source coordinate \((x_s, y_s)\) by mapping output → input.
- Round to nearest integer indices \((\text{round}(x_s), \text{round}(y_s))\).

**E. Without it**  
You may default to more complex methods even if they are not necessary.

**F. Expected results**  
Blocky/jagged edges, but shape preserved.

**G. Intuition**  
“Pick the closest original pixel, don’t invent new intermediate values.”

#### 2.5.2 Bilinear

**A. What it does**  
Blends the 4 nearest neighbors around \((x_s, y_s)\) using linear interpolation in both x and y.

**B. Why**  
Produces smoother results than nearest neighbor, with moderate cost.

**C. When**  
Typical choice for image resizing when quality is important but speed still matters.

**D. Logic**
- Let \(x_0 = \lfloor x_s \rfloor, y_0 = \lfloor y_s \rfloor\).
- Use:
  \[
  f(x_s, y_s) \approx
  (1-dx)(1-dy) f(x_0,y_0) +
  dx(1-dy) f(x_0+1,y_0) +
  (1-dx)dy f(x_0,y_0+1) +
  dx\,dy f(x_0+1,y_0+1)
  \]

**E. Without it**  
Nearest neighbor may be too blocky; bicubic may be too slow for large images.

**F. Expected results**  
Relatively smooth resized images, slightly blurred but visually pleasant.

**G. Intuition**  
“Take a weighted average of the four closest pixels proportional to distance.”

#### 2.5.3 Bicubic

**A. What it does**  
Uses **cubic interpolation** in both axes with a kernel (Catmull-Rom) based on 16 neighbors.

**B. Why**  
Provides smoother and sharper resizing than bilinear, at higher computational cost.

**C. When**  
High-quality resizing, printing, or when artifacts must be minimized.

**D. Logic**
- Use cubic weighting function \(w(t)\) with parameter \(a = -0.5\).
- For each axis, combine four neighbors (from \(-1\) to \(+2\) around floor index).
- Final value is weighted sum of 16 pixels.

**E. Without it**  
Fine structures may look less sharp; bilinear may blur too much at large scale changes.

**F. Expected results**  
Sharp but smooth image with fewer aliasing artifacts than bilinear.

**G. Intuition**
- **Simple**: Smooth curves instead of straight lines in interpolation.  
- **Math**: Cubic polynomials create \(C^1\) continuity (continuous first derivative), so transitions are smoother.

---

### 2.6 Cropping

**A. What it does**  
Extracts a rectangular **region of interest (ROI)** from the image.

**B. Why**  
To focus on relevant parts, reduce noise, or prepare inputs for further processing.

**C. When**  
Before running filters or compression on a specific object; to reduce input size.

**D. Logic**
- User specifies coordinates \((x_1, y_1, x_2, y_2)\), or drags on the preview canvas.
- These are mapped to image coordinates using zoom factor.
- The ROI is `image[y1:y2, x1:x2]`.

**E. Without it**  
You process the entire image, possibly wasting computation and including irrelevant regions.

**F. Expected results**  
The cropped part appears as a new preview image, dimensions reduced.

**G. Intuition**  
Just taking a submatrix (sub-rectangle) of the original pixel matrix.

---

### 2.7 Histogram Analysis

#### 2.7.1 Histogram Calculation

**A. What it does**  
Counts how many pixels fall into each intensity value (0–255) for a grayscale image.

**B. Why**  
Histogram describes **global brightness distribution** and is essential for:
- Contrast analysis
- Thresholding decisions
- Equalization

**C. When**  
Whenever you want to analyze exposure, contrast, or prepare for equalization.

**D. Logic**
- Convert to grayscale.
- Initialize array `hist[0..255] = 0`.
- For each pixel value `v`, increment `hist[v]`.
- Display as a bar graph.

**E. Without it**  
You cannot reason quantitatively about brightness and contrast.

**F. Expected results**  
256-length vector; the plotted histogram shows which intensity ranges dominate.

**G. Intuition**  
Think of it as a **frequency table** of brightness levels.

#### 2.7.2 Good vs Bad Histogram Reasoning

**A. What it does**  
Classifies the histogram as good/bad/underexposed/overexposed based on:
- Spread (how many bins are non-zero)
- Fractions of pixels at very low or very high intensities

**B. Why**  
Gives qualitative feedback on contrast and exposure to the user.

**C. When**  
After computing histogram, before/noticing whether equalization might help.

**D. Logic**
- Compute:
  - `spread = count_nonzero(hist)`
  - `low = sum(hist[0:32]) / total`
  - `high = sum(hist[224:256]) / total`
- Reasoning:
  - **High spread, low low/high** → good contrast.
  - **Low spread** → low contrast.
  - **Large low** → underexposed (too dark).
  - **Large high** → overexposed (too bright).

**E. Without it**  
User might not understand whether the image needs improvement.

**F. Expected results**  
A short human-readable explanation: “well-spread”, “underexposed”, etc.

**G. Intuition**  
You’re checking whether the histogram is **concentrated** or **spread out** and whether it’s biased to dark or bright ends.

#### 2.7.3 Histogram Equalization

**A. What it does**  
Re-distributes intensities to **spread** the histogram, improving contrast.

**B. Why**  
Enhances low-contrast images where intensities occupy a small range.

**C. When**  
When histogram shows a narrow range; image looks dull or washed out.

**D. Logic**
1. Compute histogram and CDF (cumulative distribution function) of intensities.
2. Normalize CDF to [0, 255] to create a **look-up table (LUT)**.
3. Map each pixel \(v\) to new value \(LUT[v]\).

Mathematically, if \(F(v)\) is CDF,
\[
v_{\text{new}} = \text{round}(255 \cdot F(v))
\]

**E. Without it**  
Low-contrast images remain difficult to interpret; important details may be hidden.

**F. Expected results**  
Balanced image with better visibility of details in both dark and bright regions.

**G. Intuition**  
Equalization tries to make the histogram more “flat” → equal use of the dynamic range.

---

### 2.8 Low-Pass Filters

#### 2.8.1 Gaussian Filter (19×19, σ = 3)

**A. What it does**  
Blurs the image using a large Gaussian kernel to reduce **high-frequency noise**.

**B. Why**  
Good for smoothing while minimizing artifacts; often used before edge detection or downsampling.

**C. When**  
To remove fine-grain noise or details; pre-processing before compression or segmentation.

**D. Logic**
- Kernel:
  \[
  G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}
  \]
- Build a 19×19 kernel with \(\sigma = 3\), normalize so sum = 1.
- Convolve kernel with image (nested loops over x, y).

**E. Without it**  
Noise remains; compression may be less efficient; edge detectors may produce spurious edges.

**F. Expected results**  
Smooth, slightly blurred image; noise reduced but edges a bit weaker.

**G. Intuition**  
Each pixel becomes a **weighted average** of neighbors with weights that decrease with distance from center.

#### 2.8.2 Median Filter (7×7)

**A. What it does**  
Replaces each pixel by the **median** of its 7×7 neighborhood.

**B. Why**  
Excellent for removing **impulse (salt-and-pepper)** noise while preserving edges.

**C. When**  
When you see isolated white/black dots; after transmission errors or sensor spikes.

**D. Logic**
- For each pixel:
  - Extract 7×7 window.
  - Sort intensities and pick middle value.

**E. Without it**  
Noise spikes stay; Gaussian smoothing would blur them but also blur edges more.

**F. Expected results**  
Cleaner image with edges largely preserved but isolated extreme pixels removed.

**G. Intuition**  
Median is robust: extreme values don’t affect it as much as the mean.

---

### 2.9 High-Pass Filters

#### 2.9.1 Laplacian

**A. What it does**  
Second-derivative operator; highlights **regions of rapid intensity change** (edges, corners).

**B. Why**  
Used to detect edges or to sharpen images when added back to original.

**C. When**  
Edge detection; blob detection; pre-processing for segmentation.

**D. Logic**
- Kernel example:
  \[
  \begin{bmatrix}
  0 & 1 & 0 \\
  1 & -4 & 1 \\
  0 & 1 & 0
  \end{bmatrix}
  \]
- Convolve with grayscale image; magnitude indicates strength of second derivative.

**E. Without it**  
You cannot capture fine transitions clearly; edges not explicitly highlighted.

**F. Expected results**  
Image dominated by edges and corners; flat regions near zero.

**G. Intuition**  
Laplacian measures how different a pixel is from its neighbors in *second-order* sense.

#### 2.9.2 Sobel

**A. What it does**  
Computes **first derivatives** in X and Y, then combines them to form an edge magnitude.

**B. Why**  
Standard gradient-based edge detector; directional edge information.

**C. When**  
Edge detection with some noise robustness.

**D. Logic**
- Kernels:
  \[
  G_x =
  \begin{bmatrix}
  -1&0&1\\
  -2&0&2\\
  -1&0&1
  \end{bmatrix},
  \quad
  G_y =
  \begin{bmatrix}
  -1&-2&-1\\
  0&0&0\\
  1&2&1
  \end{bmatrix}
  \]
- Apply both, compute:
  \[
  M = \sqrt{G_x^2 + G_y^2}
  \]

**E. Without it**  
You lose directional gradient information; edges may not be detected robustly.

**F. Expected results**  
Bright lines along edges; darker elsewhere.

**G. Intuition**  
It approximates partial derivatives \(\partial I / \partial x\) and \(\partial I / \partial y\).

#### 2.9.3 Gradient (First-derivative Filter)

**A. What it does**  
Uses simple difference kernels to approximate first derivative (e.g., [0, -1, 1]).

**B. Why**  
Simplest gradient filter; used to illustrate derivative idea.

**C. When**  
Teaching/understanding gradients; quick edge-like detection.

**D. Logic**
- Kernels:
  - Horizontal: \([0, -1, 1]\)
  - Vertical: \([0, -1, 1]^T\)
- Add absolute values to get gradient magnitude.

**E. Without it**  
You have less intuition on how derivatives work in discrete images.

**F. Expected results**  
Similar to Sobel but less smoothed, more sensitive to noise.

**G. Intuition**  
First derivative ~ difference between neighboring pixels.

---

### 2.10 Compression Techniques

In compression, we:
- Take the image (often grayscale).
- Convert to a sequence of symbols or coefficients.
- Apply a coding scheme.
- Decode to reconstruct the image.
- Compute:
  - **Compression ratio** = original bits / compressed bits.
  - **PSNR** to measure reconstruction quality.

#### 2.10.1 Huffman Coding

**A. What it does**  
Variable-length, prefix-free coding where **frequent symbols get shorter codes**.

**B. Why**  
Lossless compression; optimal for a known symbol distribution (for integer-length codes).

**C. When**  
Baseline entropy coding for grayscale images or residuals.

**D. Logic**
1. Count frequencies of pixel values (0–255).
2. Build a binary tree (min-heap): merge two lowest-frequency nodes iteratively.
3. Assign 0/1 along branches to create bit-codes.
4. Encode image as concatenation of codes.

**E. Without it**  
You can’t exploit non-uniform frequency distribution; storage stays at 8 bits/pixel.

**F. Expected results**  
Shorter bitstream for images with skewed histograms.

**G. Intuition**  
Like Morse code: common letters get short patterns, rare letters get long patterns.

#### 2.10.2 Golomb–Rice Coding

**A. What it does**  
Encodes non-negative integers based on quotient and remainder with respect to \(2^k\).

**B. Why**  
Very efficient for **geometric-like distributions** (e.g., many small values).

**C. When**  
Good for residuals or run lengths that are small with occasional larger values.

**D. Logic**
For value \(v\) and parameter \(k\):
- \(m = 2^k\)
- \(q = v // m\), \(r = v \% m\)
- Code: \(q\) ones + a zero + binary of \(r\) (k bits).

**E. Without it**  
You may use generic codes (Huffman) which might be slightly less efficient for such distributions.

**F. Expected results**  
Compact coding for many small values.

**G. Intuition**  
Split a number into “how many groups of size m” (quotient) and “remainder inside the group”.

#### 2.10.3 Arithmetic Coding

**A. What it does**  
Encodes a whole sequence into a single **fractional number** in [0,1), using symbol probabilities.

**B. Why**  
Can achieve compression rates closer to theoretical entropy than Huffman.

**C. When**  
High-compression scenarios where complexity is acceptable.

**D. Logic**
1. Compute symbol probabilities and cumulative distribution function (CDF).
2. Start with interval [0,1).
3. For each symbol, narrow interval according to CDF of that symbol.
4. Final code is any number inside the last interval.
5. Decoder runs the process in reverse.

**E. Without it**  
You’re limited to integer-bit symbol codes; might waste some bits.

**F. Expected results**  
Code length ~ entropy × number of symbols.

**G. Intuition**  
Progressively “zoom in” on an interval; each symbol choice narrows down where you are.

#### 2.10.4 LZW Coding

**A. What it does**  
Dictionary-based, lossless compression that builds dictionary of **variable-length sequences**.

**B. Why**  
Effective for images with repeated patterns; used in formats like GIF.

**C. When**  
When there are many repeated sequences of pixel values.

**D. Logic**
- Initialize dictionary with all single-byte values (0–255).
- Read input stream, grow **current string** until it’s not in dictionary.
- Output code for the longest existing string, then add new string to dictionary.

**E. Without it**  
You lose an important class of dictionary coding methods.

**F. Expected results**  
Good compression for images with repeating structures.

**G. Intuition**  
Instead of coding each pixel separately, remember and reuse recurring **chunks**.

#### 2.10.5 Run-Length Encoding (RLE)

**A. What it does**  
Encodes runs of the same value as (value, count) pairs.

**B. Why**  
Effective for long stretches of the same intensity (e.g., lines, backgrounds).

**C. When**  
For images with many constant regions, cartoons, binary images, etc.

**D. Logic**
- Traverse flattened pixels.
- If value repeats, increment count; when it changes, output (value, count).

**E. Without it**  
You waste bits storing identical values repeatedly.

**F. Expected results**  
Significant compression for images with long runs; maybe worse for very noisy images.

**G. Intuition**  
“There are 20 zeros in a row” instead of writing zero 20 times.

#### 2.10.6 Symbol-Based Coding

**A. What it does**  
Gives **short codes** to top-N most frequent symbols and fallback codes for others.

**B. Why**  
Simplified approximated entropy coding; demonstrates frequency-based coding ideas.

**C. When**  
Educational demonstrations or simple custom codecs.

**D. Logic**
- Find top-16 most frequent intensities; assign them 5-bit codes.
- All others get a 9-bit pattern (`1 + full 8-bit value`).

**E. Without it**  
Less variety in coding strategies; you rely solely on Huffman/arith/LZW.

**F. Expected results**  
Compression if there is a strong skew towards a small set of intensities.

**G. Intuition**  
“VIP symbols” get short tickets; everyone else uses a longer generic ticket.

#### 2.10.7 Bit-Plane Coding

**A. What it does**  
Separates an 8-bit grayscale image into 8 binary bitplanes (from LSB to MSB).

**B. Why**  
Higher bitplanes hold more structure; lower planes often look like noise and can be compressed aggressively.

**C. When**  
Layered compression, progressive transmission, or analysis of bit significance.

**D. Logic**
For each bit \(b \in [0,7]\):
- \( \text{plane}_b(x,y) = \frac{\text{pixel}(x,y) \,\&\, (1 \ll b)}{2^b} \)

**E. Without it**  
You cannot selectively compress or drop low-significance bits.

**F. Expected results**  
8 binary images; MSB planes show main shapes, LSB planes show finer details/noise.

**G. Intuition**  
Decomposing a number in base 2 layers.

#### 2.10.8 Block Transform Coding (DCT)

**A. What it does**  
Applies an 8×8 **Discrete Cosine Transform (DCT)** to image blocks and quantizes coefficients.

**B. Why**  
Foundation of JPEG; compacts energy into a few low-frequency coefficients.

**C. When**  
Lossy image compression with good visual quality.

**D. Logic**
1. Partition image into 8×8 blocks.
2. Subtract 128 (center intensities around 0).
3. Apply 2D DCT:
   \[
   C(u,v) = \alpha(u)\alpha(v)\sum_{x=0}^{7}\sum_{y=0}^{7}
   f(x,y)\cos\frac{(2x+1)u\pi}{16}\cos\frac{(2y+1)v\pi}{16}
   \]
4. Divide by quantization matrix Q and round.
5. Store quantized coefficients.
6. Decompress: multiply by Q and apply inverse DCT.

**E. Without it**  
You lose a powerful transform-based lossy compression; cannot exploit **frequency domain**.

**F. Expected results**  
Significant size reduction with minor visible degradation if quantization is moderate.

**G. Intuition**  
Expressing image as a sum of cosine basis patterns; most energy is in low frequencies, so many high-frequency coefficients can be coarsely quantized or set to zero.

#### 2.10.9 Predictive Coding (DPCM)

**A. What it does**  
Predicts each pixel from its neighbor (e.g., left pixel) and encodes the **difference** (residual).

**B. Why**  
Differences are usually smaller and more compressible than original samples.

**C. When**  
Lossless/near-lossless image and audio coding, predictive pipelines.

**D. Logic**
For each pixel:
\[
e(x,y) = f(x,y) - \hat{f}(x,y)
\]
where \(\hat{f}(x,y)\) is typically \(f(x-1,y)\).  
Store `e`, reconstruct by:
\[
f(x,y) = e(x,y) + \hat{f}(x,y)
\]

**E. Without it**  
You don’t exploit local correlation; entropy remains higher.

**F. Expected results**  
Residual image with values near zero; highly compressible using entropy coding.

**G. Intuition**  
Neighboring pixels are similar; it’s cheaper to encode how much they differ than their absolute values.

#### 2.10.10 Wavelet Compression (Haar)

**A. What it does**  
Applies a **Haar wavelet transform**, separating image into approximation and detail subbands, then thresholds small coefficients.

**B. Why**  
Provides multi-resolution representation; good energy compaction like DCT but with localization.

**C. When**  
Wavelet-based compression (e.g., JPEG 2000 concepts); multi-scale analysis.

**D. Logic**
- 1D Haar on rows then columns:
  - Average and difference for pairs of pixels → low-frequency (A) and high-frequency (D) coefficients.
- After transform, apply threshold: set small coefficients to zero.
- Inverse Haar reconstructs image from retained coefficients.

**E. Without it**  
You cannot demonstrate a transform-based compression alternative to DCT with multiscale properties.

**F. Expected results**  
Good compression with acceptable artifacts; coarse approximation plus some details.

**G. Intuition**  
Haar splits signal into “smooth part” and “detail part” at various scales.

---

## 3. Possible TA Questions (With Example Answers)

Below are typical questions per methodology.

### 3.1 Grayscale & Binary

- **Q (Conceptual):** Why use the specific weights 0.299, 0.587, 0.114 in grayscale conversion?  
  **A:** They approximate human luminance perception; the eye is most sensitive to green, less to red, and least to blue.

- **Q (Implementation):** How do you convert an RGB pixel to binary?  
  **A:** First compute grayscale \(Y\), then choose a threshold \(T\) (we used the global average), and set the pixel to 255 if \(Y \ge T\) else 0.

- **Q (Trick):** If an image has a very bright background and small dark object, will average threshold always work well?  
  **A:** No. The average can be biased by large bright background, causing the object to be misclassified. More advanced local/adaptive thresholds might be needed.

### 3.2 Affine Transformations

- **Q:** Why do we use backward mapping instead of forward mapping?  
  **A:** Backward mapping ensures every output pixel is assigned a value by sampling from input; forward mapping can leave gaps/holes.

- **Q:** How is bilinear interpolation involved in affine transforms?  
  **A:** When mapping output coordinates back to non-integer input coordinates, we estimate pixel value by bilinear interpolation from the four nearest neighbors.

- **Q:** Is rotation an affine transform? Why?  
  **A:** Yes. It can be represented as a linear transformation with a rotation matrix in homogeneous coordinates.

### 3.3 Interpolation

- **Q:** Compare nearest neighbor vs bilinear vs bicubic.  
  **A:** Nearest is fastest but blocky; bilinear is smoother but slightly blurred; bicubic uses more neighbors and cubic polynomials, giving smoother and sharper results at higher cost.

- **Q (Trick):** Why can resizing with nearest neighbor be problematic for photographic images?  
  **A:** It introduces blockiness and aliasing; transitions are not smooth.

### 3.4 Histogram & Equalization

- **Q:** What is a histogram and what does a narrow histogram imply?  
  **A:** It’s a frequency distribution of intensities. A narrow histogram implies low contrast, with pixel values confined to a small range.

- **Q:** How does histogram equalization enhance contrast?  
  **A:** By using the CDF to remap intensities so that their distribution is more uniform across the available range.

- **Q (Trick):** Can histogram equalization reduce contrast in some regions?  
  **A:** Yes. It may over-amplify noise or compress certain ranges if the original distribution is multi-modal.

### 3.5 Filters

- **Q:** Why is median filter better than Gaussian for salt-and-pepper noise?  
  **A:** Median is robust to outliers; it removes isolated spikes without averaging them into neighbors, preserving edges.

- **Q:** What is the main purpose of high-pass filters like Sobel or Laplacian?  
  **A:** To emphasize rapid intensity changes → edges and fine details.

- **Q (Implementation):** How did you implement convolution manually?  
  **A:** By padding the image and sliding the kernel over each position, computing the sum of element-wise products.

### 3.6 Compression

- **Q:** What is compression ratio and how did you compute it?  
  **A:** \( \text{CR} = \frac{\text{original bits}}{\text{compressed bits}} \). Original bits = number of pixels × bits/pixel; compressed bits estimated from code lengths.

- **Q:** Difference between lossless and lossy methods in this project?  
  **A:** Huffman, Golomb–Rice, Arithmetic (with discrete symbols), LZW, RLE are conceptually lossless (if fully implemented). DCT and Wavelet with thresholding/quantization are lossy.

- **Q (Trick):** Why do transform-based methods like DCT typically give better visual quality at same bitrate than pure symbol coding?  
  **A:** They operate in frequency domain where energy is concentrated in low frequencies; discarding high-frequencies affects perception less.

---

## 4. Codebase Breakdown

### 4.1 Folder Structure

```text
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

### 4.2 Module Responsibilities

- **`main.py`**  
  Entry point. Starts the Tkinter application by calling the GUI runner.

- **`gui.py`**  
  - Builds the full **Tkinter interface** (buttons, frames, canvases).
  - Handles image upload, display, preview/apply logic, theme toggle, zoom.
  - Connects each button to the corresponding algorithm function from other modules.
  - Shows compression ratios and PSNR.

- **`utils.py`**  
  - Helper functions:
    - Image loading to NumPy (via PIL).
    - Conversion from NumPy → PIL for display/saving.
    - Grayscale enforcement, normalization for display.
    - Gaussian kernel creation, generic convolution, PSNR, compression ratio.

- **`image_loader.py`**  
  - Wraps file-based image reading/saving.
  - Provides image info dictionary (width, height, size, mode).

- **`grayscale.py`**  
  - Implements manual grayscale using luminance weights.

- **`binary.py`**  
  - Computes average intensity threshold.
  - Converts grayscale to binary.
  - Evaluates whether threshold is “balanced” or biased.

- **`affine.py`**  
  - Implements translation, scaling, rotation, shear X, shear Y.
  - Uses matrices + backward mapping + bilinear interpolation.

- **`interpolation.py`**  
  - Implements nearest neighbor, bilinear, and bicubic interpolation functions.

- **`histogram.py`**  
  - Computes 256-bin histogram.
  - Evaluates histogram quality.
  - Performs histogram equalization.

- **`filters.py`**  
  - Gaussian low-pass filter (19×19, σ=3).
  - Median filter (7×7).

- **`highpass.py`**  
  - Laplacian, Sobel, and gradient filters on grayscale images.

- **`compression/` package**  
  - `huffman.py` – Huffman coding encode/decode.
  - `golomb_rice.py` – Golomb–Rice encode/decode.
  - `arithmetic.py` – Arithmetic coding.
  - `lzw.py` – LZW dictionary coding.
  - `rle.py` – Run-length coding.
  - `bitplane.py` – Bit-plane splitting and reconstruction.
  - `dct.py` – 8×8 DCT, quantization, and inverse.
  - `predictive.py` – DPCM-like predictive coding.
  - `wavelet.py` – Haar wavelet transform and thresholding.
  - `symbol_based.py` – Custom symbol coding with top-16 table.

### 4.3 How Files Interact (Execution Flow)

High-level diagram:

```text
main.py
  └── gui.run_app()
         ├── image_loader.load_image()
         ├── grayscale.to_grayscale()
         ├── binary.to_binary(), binary.threshold_evaluation()
         ├── affine.translate/scale/rotate/shear_x/shear_y()
         ├── interpolation.nearest_neighbor/bilinear/bicubic()
         ├── histogram.compute_histogram(), equalize()
         ├── filters.gaussian_blur(), median_filter()
         ├── highpass.laplacian()/sobel()/gradient()
         └── compression.*.compress()/decompress()
```

### 4.4 Example Execution Flow (Pressing a Button)

Example: **User clicks “Grayscale” button**

1. Button’s command in `gui.py` calls `do_grayscale()`.
2. `do_grayscale()`:
   - Checks if a current image is loaded.
   - Calls `grayscale.to_grayscale(self.current_img)`.
3. The result is stored as `self.preview_img`.
4. `render_images()` displays the preview next to the original.
5. If user clicks **“Apply Preview”**, `self.current_img` is updated to the preview.

Similarly, for **Huffman compression**:

1. User clicks “Huffman” button.
2. GUI converts current image to grayscale.
3. Calls `huffman.compress(gray)` → returns bitstring & codebook.
4. Calls `huffman.decompress(...)` → reconstructed image.
5. Computes compression ratio and PSNR in `gui.py`.
6. Displays reconstructed image as preview and shows ratio text.

---

## 5. Critical Mistakes Students Usually Make

- **Incorrect coordinate mapping in affine transforms**
  - Using forward mapping leads to holes.
  - Forgetting to use inverse matrix for mapping output → input.

- **Ignoring boundary conditions**
  - Not checking if mapped coordinates fall outside the image.
  - Not padding correctly for convolution.

- **Wrong grayscale formula**
  - Averaging channels instead of using proper luminance weights.

- **Misinterpreting histograms**
  - Thinking “flat histogram” is always good, even for images that don’t need it.
  - Ignoring that equalization can over-amplify noise.

- **Poor thresholding assumptions**
  - Assuming global mean threshold always gives good segmentation.

- **Filter implementation mistakes**
  - Mixing up x/y loops or kernel indices.
  - Forgetting to normalize Gaussian kernel.

- **Compression confusion**
  - Mixing up lossless vs lossy methods.
  - Not understanding that DCT quantization is where loss occurs.
  - Forgetting that predictive coding must use identical prediction in encoder & decoder.

- **Performance issues**
  - Using Python loops for everything without NumPy where possible (but here, manual loops are intentionally used for learning).
  - Applying huge kernels on very large images without realizing time cost.

> **TA Focus:**  
> - Do you understand **why** each method is used, not just how to call it?  
> - Can you explain trade-offs (quality vs speed, lossless vs lossy)?  
> - Can you reason about edge cases (e.g., border handling, extreme histograms, noise types)?

---

## 6. Final Summary

By studying this project, you should be able to:

- Explain and implement **core image-processing operations**: grayscale, binary thresholding, histogram-based analysis, spatial filters, and affine transforms.
- Understand and compare **interpolation methods** and their visual impacts.
- Describe **low-pass vs high-pass** filters and when to use each.
- Explain and demonstrate a full range of **compression techniques**, from entropy and dictionary coding to transform and predictive/wavelet methods.
- Navigate the **code structure**, following data from `main.py` → GUI callbacks → algorithm modules → back to the GUI display.
- Defend design decisions in a viva:
  - Why use backward mapping for transforms?
  - Why specific thresholds or kernel sizes?
  - Why some methods are lossy and when that is acceptable.

Together, these components form a **complete educational pipeline**: acquisition, processing, analysis, enhancement, and compression of digital images, all implemented manually for maximum learning value.


