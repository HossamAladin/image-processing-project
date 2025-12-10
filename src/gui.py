import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk, ImageDraw

from . import grayscale, binary, affine, interpolation, histogram, filters, highpass
from .image_loader import load_image, save_image
from .utils import array_to_pil, ensure_grayscale, normalize_image, compression_ratio, psnr
from .compression import (
    huffman,
    lzw,
    arithmetic,
    rle,
    golomb_rice,
    bitplane,
    dct,
    predictive,
    wavelet,
    symbol_based,
)


class ImageApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Manual Image Processing Suite")
        self.theme = "light"
        self.bg_light = "#f7f7fb"
        self.bg_dark = "#1e1e1e"
        self.fg_light = "#1e1e1e"
        self.fg_dark = "#e5e5e5"
        self.accent = "#5b8def"

        self.original_img = None
        self.current_img = None
        self.preview_img = None
        self.original_info = {}
        self.zoom_factor = 1.0
        self.hist_canvas = None
        self.selection = None
        self.start_sel = None
        self.end_sel = None

        self._build_ui()
        self._apply_theme()

    # --- UI helpers ---
    def _build_ui(self):
        self.root.geometry("1200x800")
        self.root.configure(bg=self.bg_light)

        top = tk.Frame(self.root)
        top.pack(fill=tk.X, padx=10, pady=6)

        tk.Button(top, text="Upload Image", command=self.load_image).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Save Result", command=self.save_current).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Apply Preview", command=self.apply_preview).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Reset", command=self.reset_image).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Theme Toggle", command=self.toggle_theme).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Zoom +", command=lambda: self.change_zoom(1.2)).pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Zoom -", command=lambda: self.change_zoom(0.8)).pack(side=tk.LEFT, padx=4)

        self.info_label = tk.Label(top, text="No image loaded.")
        self.info_label.pack(side=tk.LEFT, padx=10)

        body = tk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        # Image display area (left side)
        img_frame = tk.Frame(body)
        img_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Label(img_frame, text="Original").pack()
        self.canvas_original = tk.Canvas(img_frame, width=400, height=350, bg="#ddd")
        self.canvas_original.pack(padx=5, pady=5)

        tk.Label(img_frame, text="Current / Preview").pack()
        self.canvas_result = tk.Canvas(img_frame, width=400, height=350, bg="#ddd")
        self.canvas_result.pack(padx=5, pady=5)
        self.canvas_result.bind("<ButtonPress-1>", self.start_selection)
        self.canvas_result.bind("<B1-Motion>", self.update_selection)
        self.canvas_result.bind("<ButtonRelease-1>", self.finish_selection)

        # Right control panel (scrollable column of sections)
        control_container = tk.Frame(body)
        control_container.pack(side=tk.RIGHT, fill=tk.Y)

        self.control_canvas = tk.Canvas(
            control_container,
            highlightthickness=0,
            borderwidth=0,
        )
        scrollbar = tk.Scrollbar(control_container, orient="vertical", command=self.control_canvas.yview)
        self.control_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.control_canvas.pack(side=tk.LEFT, fill=tk.Y)

        control = tk.Frame(self.control_canvas)
        self.control_canvas.create_window((0, 0), window=control, anchor="nw")

        def _update_scrollregion(event):
            self.control_canvas.configure(scrollregion=self.control_canvas.bbox("all"))

        control.bind("<Configure>", _update_scrollregion)

        self._section_basic(control)
        self._section_affine(control)
        self._section_interp(control)
        self._section_hist(control)
        self._section_filters(control)
        self._section_highpass(control)
        self._section_compress(control)
        self._section_crop(control)

    def _section_basic(self, parent):
        frame = self._section_frame(parent, "Basic")
        tk.Button(frame, text="Grayscale", command=self.do_grayscale).pack(fill=tk.X, pady=2)
        tk.Button(frame, text="Binary", command=self.do_binary).pack(fill=tk.X, pady=2)

    def _section_affine(self, parent):
        frame = self._section_frame(parent, "Affine")
        self.entry_tx = self._labeled_entry(frame, "Tx", "20")
        self.entry_ty = self._labeled_entry(frame, "Ty", "20")
        tk.Button(frame, text="Translate", command=self.do_translate).pack(fill=tk.X, pady=2)
        self.entry_sx = self._labeled_entry(frame, "Sx", "1.2")
        self.entry_sy = self._labeled_entry(frame, "Sy", "1.2")
        tk.Button(frame, text="Scale", command=self.do_scale).pack(fill=tk.X, pady=2)
        self.entry_rot = self._labeled_entry(frame, "Angle", "30")
        tk.Button(frame, text="Rotate", command=self.do_rotate).pack(fill=tk.X, pady=2)
        self.entry_shx = self._labeled_entry(frame, "Shear X", "0.2")
        tk.Button(frame, text="Shear X", command=self.do_shear_x).pack(fill=tk.X, pady=2)
        self.entry_shy = self._labeled_entry(frame, "Shear Y", "0.2")
        tk.Button(frame, text="Shear Y", command=self.do_shear_y).pack(fill=tk.X, pady=2)

    def _section_interp(self, parent):
        frame = self._section_frame(parent, "Interpolation")
        self.entry_w = self._labeled_entry(frame, "Width", "300")
        self.entry_h = self._labeled_entry(frame, "Height", "300")
        tk.Button(frame, text="Nearest", command=lambda: self.do_resize("nearest")).pack(fill=tk.X, pady=2)
        tk.Button(frame, text="Bilinear", command=lambda: self.do_resize("bilinear")).pack(fill=tk.X, pady=2)
        tk.Button(frame, text="Bicubic", command=lambda: self.do_resize("bicubic")).pack(fill=tk.X, pady=2)

    def _section_hist(self, parent):
        frame = self._section_frame(parent, "Histogram")
        tk.Button(frame, text="Compute Histogram", command=self.show_histogram).pack(fill=tk.X, pady=2)
        tk.Button(frame, text="Equalize", command=self.do_equalize).pack(fill=tk.X, pady=2)
        self.hist_eval_label = tk.Label(frame, text="Evaluation: -", wraplength=180, justify="left")
        self.hist_eval_label.pack(fill=tk.X, pady=2)

    def _section_filters(self, parent):
        frame = self._section_frame(parent, "Low-Pass Filters")
        tk.Button(frame, text="Gaussian 19x19 σ=3", command=self.do_gaussian).pack(fill=tk.X, pady=2)
        tk.Button(frame, text="Median 7x7", command=self.do_median).pack(fill=tk.X, pady=2)

    def _section_highpass(self, parent):
        frame = self._section_frame(parent, "High-Pass Filters")
        tk.Button(frame, text="Laplacian", command=self.do_laplacian).pack(fill=tk.X, pady=2)
        tk.Button(frame, text="Sobel", command=self.do_sobel).pack(fill=tk.X, pady=2)
        tk.Button(frame, text="Gradient", command=self.do_gradient).pack(fill=tk.X, pady=2)

    def _section_compress(self, parent):
        frame = self._section_frame(parent, "Compression")
        self.compress_status = tk.Label(frame, text="Ratio: -")
        self.compress_status.pack(fill=tk.X, pady=2)
        tk.Button(frame, text="Huffman", command=self.do_huffman).pack(fill=tk.X, pady=1)
        tk.Button(frame, text="Golomb-Rice", command=self.do_golomb).pack(fill=tk.X, pady=1)
        tk.Button(frame, text="Arithmetic", command=self.do_arithmetic).pack(fill=tk.X, pady=1)
        tk.Button(frame, text="LZW", command=self.do_lzw).pack(fill=tk.X, pady=1)
        tk.Button(frame, text="RLE", command=self.do_rle).pack(fill=tk.X, pady=1)
        tk.Button(frame, text="Symbol Coding", command=self.do_symbol).pack(fill=tk.X, pady=1)
        tk.Button(frame, text="Bit-Plane", command=self.do_bitplane).pack(fill=tk.X, pady=1)
        tk.Button(frame, text="DCT 8x8", command=self.do_dct).pack(fill=tk.X, pady=1)
        tk.Button(frame, text="Predictive", command=self.do_predictive).pack(fill=tk.X, pady=1)
        tk.Button(frame, text="Wavelet", command=self.do_wavelet).pack(fill=tk.X, pady=1)

    def _section_crop(self, parent):
        frame = self._section_frame(parent, "Crop")
        self.entry_x1 = self._labeled_entry(frame, "x1", "0")
        self.entry_y1 = self._labeled_entry(frame, "y1", "0")
        self.entry_x2 = self._labeled_entry(frame, "x2", "100")
        self.entry_y2 = self._labeled_entry(frame, "y2", "100")
        tk.Button(frame, text="Crop by coords", command=self.do_crop_coords).pack(fill=tk.X, pady=2)
        tk.Label(frame, text="Or drag on preview canvas to crop").pack(fill=tk.X, pady=2)

    def _section_frame(self, parent, title: str):
        frame = tk.LabelFrame(parent, text=title, padx=6, pady=4)
        frame.pack(fill=tk.X, pady=4)
        return frame

    def _labeled_entry(self, frame, text, default):
        row = tk.Frame(frame)
        row.pack(fill=tk.X, pady=1)
        tk.Label(row, text=text, width=8, anchor="w").pack(side=tk.LEFT)
        entry = tk.Entry(row, width=8)
        entry.insert(0, default)
        entry.pack(side=tk.RIGHT)
        return entry

    def _apply_theme(self):
        bg = self.bg_dark if self.theme == "dark" else self.bg_light
        fg = self.fg_dark if self.theme == "dark" else self.fg_light
        self.root.configure(bg=bg)
        for widget in self.root.winfo_children():
            self._apply_theme_recursive(widget, bg, fg)

    def _apply_theme_recursive(self, widget, bg, fg):
        try:
            widget.configure(bg=bg, fg=fg)
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            self._apply_theme_recursive(child, bg, fg)

    # --- Core actions ---
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not path:
            return
        arr, info = load_image(path)
        self.original_img = arr
        self.current_img = arr.copy()
        self.preview_img = None
        self.original_info = info
        self.update_info()
        self.render_images()

    def save_current(self):
        if self.current_img is None:
            messagebox.showerror("No image", "Load an image first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if not path:
            return
        save_image(self.current_img, path)
        messagebox.showinfo("Saved", f"Saved to {path}")

    def apply_preview(self):
        if self.preview_img is not None:
            self.current_img = self.preview_img.copy()
            self.preview_img = None
            self.render_images()

    def reset_image(self):
        if self.original_img is not None:
            self.current_img = self.original_img.copy()
            self.preview_img = None
            self.render_images()

    def toggle_theme(self):
        self.theme = "dark" if self.theme == "light" else "light"
        self._apply_theme()

    def change_zoom(self, factor: float):
        self.zoom_factor = max(0.2, min(3.0, self.zoom_factor * factor))
        self.render_images()

    def update_info(self):
        if not self.original_img is None:
            info = self.original_info
            self.info_label.config(
                text=f"{info['width']}x{info['height']} | {info['size_bytes']} bytes | {info['mode']}"
            )

    def render_images(self):
        if self.original_img is not None:
            self._render_on_canvas(self.original_img, self.canvas_original)
        target = self.preview_img if self.preview_img is not None else self.current_img
        if target is not None:
            self._render_on_canvas(target, self.canvas_result)

    def _render_on_canvas(self, img_arr: np.ndarray, canvas: tk.Canvas):
        pil_img = array_to_pil(img_arr)
        # Apply zoom for display
        w, h = pil_img.size
        disp_w, disp_h = int(w * self.zoom_factor), int(h * self.zoom_factor)
        pil_img = pil_img.resize((max(1, disp_w), max(1, disp_h)), Image.NEAREST)
        photo = ImageTk.PhotoImage(pil_img)
        canvas.image = photo  # keep reference
        canvas.delete("all")
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.config(width=photo.width(), height=photo.height())

    # --- Operations ---
    def require_image(self):
        if self.current_img is None:
            messagebox.showerror("No image", "Load an image first.")
            return False
        return True

    def set_preview(self, arr: np.ndarray):
        self.preview_img = np.clip(arr, 0, 255).astype(np.float32)
        self.render_images()

    def do_grayscale(self):
        if not self.require_image():
            return
        self.set_preview(grayscale.to_grayscale(self.current_img))

    def do_binary(self):
        if not self.require_image():
            return
        gray = ensure_grayscale(self.current_img)
        threshold = binary.threshold_average(gray)
        result = binary.to_binary(gray)
        eval_text = binary.threshold_evaluation(gray, threshold)
        self.hist_eval_label.config(text=f"Threshold {threshold:.2f} -> {eval_text}")
        self.set_preview(result)

    def do_translate(self):
        if not self.require_image():
            return
        tx = float(self.entry_tx.get())
        ty = float(self.entry_ty.get())
        self.set_preview(affine.translate(self.current_img, tx, ty))

    def do_scale(self):
        if not self.require_image():
            return
        sx = float(self.entry_sx.get())
        sy = float(self.entry_sy.get())
        self.set_preview(affine.scale(self.current_img, sx, sy))

    def do_rotate(self):
        if not self.require_image():
            return
        ang = float(self.entry_rot.get())
        self.set_preview(affine.rotate(self.current_img, ang))

    def do_shear_x(self):
        if not self.require_image():
            return
        shx = float(self.entry_shx.get())
        self.set_preview(affine.shear_x(self.current_img, shx))

    def do_shear_y(self):
        if not self.require_image():
            return
        shy = float(self.entry_shy.get())
        self.set_preview(affine.shear_y(self.current_img, shy))

    def do_resize(self, method: str):
        if not self.require_image():
            return
        w = int(self.entry_w.get())
        h = int(self.entry_h.get())
        if method == "nearest":
            res = interpolation.nearest_neighbor(self.current_img, w, h)
        elif method == "bilinear":
            res = interpolation.bilinear(self.current_img, w, h)
        else:
            res = interpolation.bicubic(self.current_img, w, h)
        self.set_preview(res)

    def show_histogram(self):
        if not self.require_image():
            return
        hist = histogram.compute_histogram(self.current_img)
        eval_text = histogram.histogram_evaluation(hist)
        self.hist_eval_label.config(text=eval_text)
        self._draw_histogram(hist)

    def _draw_histogram(self, hist: np.ndarray):
        w, h = 256, 150
        img = Image.new("RGB", (w, h), (30, 30, 30))
        draw = ImageDraw.Draw(img)
        max_v = max(hist)
        for x, v in enumerate(hist):
            bar_h = int((v / max_v) * (h - 10)) if max_v > 0 else 0
            draw.line((x, h, x, h - bar_h), fill=(90, 200, 250))
        photo = ImageTk.PhotoImage(img)
        if self.hist_canvas is None:
            self.hist_canvas = tk.Label(self.root, image=photo)
            self.hist_canvas.image = photo
            self.hist_canvas.pack(side=tk.BOTTOM, pady=4)
        else:
            self.hist_canvas.configure(image=photo)
            self.hist_canvas.image = photo

    def do_equalize(self):
        if not self.require_image():
            return
        self.set_preview(histogram.equalize(self.current_img))

    def do_gaussian(self):
        if not self.require_image():
            return
        self.set_preview(filters.gaussian_blur(self.current_img))

    def do_median(self):
        if not self.require_image():
            return
        self.set_preview(filters.median_filter(self.current_img))

    def do_laplacian(self):
        if not self.require_image():
            return
        self.set_preview(highpass.laplacian(self.current_img))

    def do_sobel(self):
        if not self.require_image():
            return
        self.set_preview(highpass.sobel(self.current_img))

    def do_gradient(self):
        if not self.require_image():
            return
        self.set_preview(highpass.gradient(self.current_img))

    # --- Compression handlers ---
    def _compression_wrapper(self, algo_name: str, recon: np.ndarray, compressed_bits: int):
        orig_bits = self.current_img.shape[0] * self.current_img.shape[1] * 8
        ratio = compression_ratio(orig_bits, compressed_bits)
        quality = psnr(ensure_grayscale(self.current_img), ensure_grayscale(recon))
        self.compress_status.config(text=f"{algo_name}: ratio {ratio:.2f} | PSNR {quality:.2f} dB")
        self.set_preview(recon)

    def do_huffman(self):
        if not self.require_image():
            return
        gray = ensure_grayscale(self.current_img)
        bits, codes = huffman.compress(gray)
        recon = huffman.decompress(bits, codes, gray.shape)
        self._compression_wrapper("Huffman", recon, len(bits))

    def do_golomb(self, k: int = 2):
        if not self.require_image():
            return
        gray = ensure_grayscale(self.current_img)
        bits, k_used, shape = golomb_rice.compress(gray, k)
        recon = golomb_rice.decompress(bits, k_used, shape)
        self._compression_wrapper("Golomb-Rice", recon, len(bits))

    def do_arithmetic(self):
        if not self.require_image():
            return
        gray = ensure_grayscale(self.current_img)
        code, probs, shape = arithmetic.compress(gray)
        recon = arithmetic.decompress(code, probs, shape)
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        bits = int(entropy * gray.size)
        self._compression_wrapper("Arithmetic", recon, bits)

    def do_lzw(self):
        if not self.require_image():
            return
        gray = ensure_grayscale(self.current_img)
        codes, _ = lzw.compress(gray)
        max_code = max(codes) if codes else 0
        bits = len(codes) * max(9, int(np.ceil(np.log2(max_code + 1))))
        recon = lzw.decompress(codes, gray.shape)
        self._compression_wrapper("LZW", recon, bits)

    def do_rle(self):
        if not self.require_image():
            return
        gray = ensure_grayscale(self.current_img)
        encoded, shape = rle.compress(gray)
        # bits: value(8) + count(16) per run
        bits = len(encoded) * (8 + 16)
        recon = rle.decompress(encoded, shape)
        self._compression_wrapper("RLE", recon, bits)

    def do_symbol(self):
        if not self.require_image():
            return
        gray = ensure_grayscale(self.current_img)
        bits, top16, shape = symbol_based.compress(gray)
        recon = symbol_based.decompress(bits, top16, shape)
        self._compression_wrapper("Symbol", recon, len(bits))

    def do_bitplane(self):
        if not self.require_image():
            return
        gray = ensure_grayscale(self.current_img)
        planes, shape = bitplane.compress(gray)
        bits = sum(len(p) for p in planes)
        recon = bitplane.decompress(planes, shape)
        self._compression_wrapper("Bit-Plane", recon, bits)

    def do_dct(self):
        if not self.require_image():
            return
        gray = ensure_grayscale(self.current_img)
        quantized, orig_shape, padded = dct.compress(gray)
        # Estimate bits from sparsity
        nonzero = sum(np.count_nonzero(q) for q in quantized)
        bits = nonzero * 12  # approximate coefficient size
        recon = dct.decompress(quantized, orig_shape, padded)
        self._compression_wrapper("DCT", recon, bits)

    def do_predictive(self):
        if not self.require_image():
            return
        gray = ensure_grayscale(self.current_img)
        diff, shape = predictive.compress(gray)
        bits = diff.size * 8  # still differences but show baseline
        recon = predictive.decompress(diff, shape)
        self._compression_wrapper("Predictive", recon, bits)

    def do_wavelet(self):
        if not self.require_image():
            return
        gray = ensure_grayscale(self.current_img)
        compressed, shape = wavelet.compress(gray)
        bits = np.count_nonzero(compressed) * 12
        recon = wavelet.decompress(compressed, shape)
        self._compression_wrapper("Wavelet", recon, bits)

    # --- Crop ---
    def do_crop_coords(self):
        if not self.require_image():
            return
        x1 = int(self.entry_x1.get())
        y1 = int(self.entry_y1.get())
        x2 = int(self.entry_x2.get())
        y2 = int(self.entry_y2.get())
        self.crop_and_preview(x1, y1, x2, y2)

    def crop_and_preview(self, x1, y1, x2, y2):
        img = self.current_img
        h, w = img.shape[:2]
        x1, x2 = sorted([max(0, x1), min(w, x2)])
        y1, y2 = sorted([max(0, y1), min(h, y2)])
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            messagebox.showerror("Crop", "Invalid crop region.")
            return
        self.set_preview(img[y1:y2, x1:x2])

    # --- Mouse selection for crop ---
    def start_selection(self, event):
        if self.current_img is None:
            return
        self.start_sel = (event.x, event.y)
        self.end_sel = (event.x, event.y)
        self._draw_selection()

    def update_selection(self, event):
        if self.start_sel is None:
            return
        self.end_sel = (event.x, event.y)
        self._draw_selection()

    def finish_selection(self, event):
        if self.start_sel is None:
            return
        self.end_sel = (event.x, event.y)
        self._draw_selection(final=True)
        x1, y1 = self.start_sel
        x2, y2 = self.end_sel
        # Map canvas coords to image coords using zoom
        inv_zoom = 1.0 / self.zoom_factor
        self.crop_and_preview(int(x1 * inv_zoom), int(y1 * inv_zoom), int(x2 * inv_zoom), int(y2 * inv_zoom))
        self.start_sel = None
        self.end_sel = None

    def _draw_selection(self, final=False):
        self.canvas_result.delete("selection")
        if self.start_sel and self.end_sel:
            x1, y1 = self.start_sel
            x2, y2 = self.end_sel
            self.canvas_result.create_rectangle(x1, y1, x2, y2, outline=self.accent, dash=(4, 2), tags="selection")
            if final:
                self.canvas_result.after(150, lambda: self.canvas_result.delete("selection"))


def run_app():
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()


