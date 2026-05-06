"""
Cuticle Melanization Analyzer
==============================
Overhead-image analysis tool for caterpillar cuticle melanization.

Usage:
    python melanization_analyzer.py

Requirements:
    pip install -r requirements.txt
"""

import tkinter as tk 
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import image_processing as ip

COLORMAPS = ["inferno", "plasma", "magma", "hot", "viridis", "jet", "YlOrRd"]
IMAGE_EXTS = ip.IMAGE_EXTS


class MelanizationApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Cuticle Melanization Analyzer")
        self.root.geometry("1280x820")
        self.root.minsize(960, 640)

        # ── Analysis state ────────────────────────────────────────────────────
        self.image_path: Path | None = None
        self.orig_image: np.ndarray | None = None
        self.gray_image: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.contour: np.ndarray | None = None
        self.centerline: np.ndarray | None = None
        self.heatmap: np.ndarray | None = None       # RGB
        self.body_intensity: float | None = None
        self.cl_intensity: float | None = None
        self.cl_profile: np.ndarray | None = None
        self.folder_path: Path | None = None

        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_menu()

        outer = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH, expand=True, padx=4, pady=(2, 0))

        left = ttk.Frame(outer)
        outer.add(left, weight=4)
        self._build_image_panel(left)

        right = ttk.Frame(outer, width=300)
        right.pack_propagate(False)
        outer.add(right, weight=1)
        self._build_controls(right)

        self.status_var = tk.StringVar(value="Ready — open an image to begin.")
        ttk.Label(
            self.root, textvariable=self.status_var,
            relief=tk.SUNKEN, anchor=tk.W, padding=(6, 2),
        ).pack(side=tk.BOTTOM, fill=tk.X)

    def _build_menu(self):
        bar = tk.Menu(self.root)

        fm = tk.Menu(bar, tearoff=0)
        fm.add_command(label="Open Image…",   command=self.cmd_load_image,  accelerator="Ctrl+O")
        fm.add_command(label="Open Folder…",  command=self.cmd_load_folder, accelerator="Ctrl+Shift+O")
        fm.add_separator()
        fm.add_command(label="Export Results…", command=self.cmd_export)
        fm.add_separator()
        fm.add_command(label="Quit", command=self.root.quit)
        bar.add_cascade(label="File", menu=fm)

        hm = tk.Menu(bar, tearoff=0)
        hm.add_command(label="About", command=self._show_about)
        bar.add_cascade(label="Help", menu=hm)

        self.root.config(menu=bar)
        self.root.bind("<Control-o>", lambda _: self.cmd_load_image())
        self.root.bind("<Control-O>", lambda _: self.cmd_load_folder())

    def _build_image_panel(self, parent):
        # View mode toggle
        vf = ttk.Frame(parent)
        vf.pack(fill=tk.X, padx=6, pady=(4, 0))
        ttk.Label(vf, text="View:").pack(side=tk.LEFT)
        self.view_var = tk.StringVar(value="Original")
        for label in ("Original", "Grayscale", "Heatmap", "Heatmap + Overlay"):
            ttk.Radiobutton(
                vf, text=label, variable=self.view_var,
                value=label, command=self._refresh_canvas,
            ).pack(side=tk.LEFT, padx=6)

        # Matplotlib canvas
        self.fig = Figure(dpi=96)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        self.fig.patch.set_facecolor("#ececec")

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        NavigationToolbar2Tk(self.canvas, parent).update()

    def _build_controls(self, parent):
        # Scrollable control panel
        canvas = tk.Canvas(parent, highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = ttk.Frame(canvas)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _resize(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(win_id, width=e.width)

        inner.bind("<Configure>", _resize)
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win_id, width=e.width))

        # Mouse-wheel scroll
        def _scroll(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _scroll)

        inner.columnconfigure(0, weight=1)
        self._build_load_section(inner)
        self._build_detection_section(inner)
        self._build_results_section(inner)
        self._build_export_section(inner)
        self._build_batch_section(inner)

    def _build_load_section(self, parent):
        f = ttk.LabelFrame(parent, text="Load", padding=8)
        f.grid(row=0, column=0, sticky="ew", padx=8, pady=4)
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)

        ttk.Button(f, text="Open Image…",  command=self.cmd_load_image
                   ).grid(row=0, column=0, sticky="ew", padx=2)
        ttk.Button(f, text="Open Folder…", command=self.cmd_load_folder
                   ).grid(row=0, column=1, sticky="ew", padx=2)

        self.file_label = ttk.Label(f, text="No file loaded", foreground="gray",
                                    wraplength=240, justify=tk.LEFT)
        self.file_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def _build_detection_section(self, parent):
        f = ttk.LabelFrame(parent, text="Detection Settings", padding=8)
        f.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        f.columnconfigure(1, weight=1)

        # Threshold
        ttk.Label(f, text="Threshold:").grid(row=0, column=0, sticky="w")
        self.thresh_var = tk.IntVar(value=0)
        self.thresh_label = ttk.Label(f, text="Auto (Otsu)", width=11, anchor="e")
        self.thresh_label.grid(row=0, column=2, sticky="e")
        ttk.Scale(
            f, from_=0, to=254, orient=tk.HORIZONTAL,
            variable=self.thresh_var, command=self._on_thresh_change,
        ).grid(row=0, column=1, sticky="ew", padx=4)

        # Colormap
        ttk.Label(f, text="Colormap:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.cmap_var = tk.StringVar(value="inferno")
        cb = ttk.Combobox(f, textvariable=self.cmap_var, values=COLORMAPS,
                          state="readonly", width=12)
        cb.grid(row=1, column=1, columnspan=2, sticky="ew", padx=4, pady=(6, 0))
        cb.bind("<<ComboboxSelected>>", lambda _: self._refresh_heatmap())

        # Smoothing
        ttk.Label(f, text="CL Smooth:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.smooth_var = tk.DoubleVar(value=1.0)
        ttk.Scale(
            f, from_=0.1, to=5.0, orient=tk.HORIZONTAL, variable=self.smooth_var,
        ).grid(row=2, column=1, columnspan=2, sticky="ew", padx=4, pady=(6, 0))

        ttk.Button(f, text="Process Image", command=self.cmd_process
                   ).grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 0))

    def _build_results_section(self, parent):
        f = ttk.LabelFrame(parent, text="Results", padding=8)
        f.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        f.columnconfigure(1, weight=1)

        bold = ("TkDefaultFont", 10, "bold")

        ttk.Label(f, text="Whole body mean:").grid(row=0, column=0, sticky="w")
        self.body_int_var = tk.StringVar(value="—")
        ttk.Label(f, textvariable=self.body_int_var, foreground="#1a3a8f",
                  font=bold).grid(row=0, column=1, sticky="e")

        ttk.Label(f, text="Dorsal CL mean:").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.cl_int_var = tk.StringVar(value="—")
        ttk.Label(f, textvariable=self.cl_int_var, foreground="#1a3a8f",
                  font=bold).grid(row=1, column=1, sticky="e", pady=(4, 0))

        ttk.Label(
            f, text="Lower value = darker = more melanized",
            foreground="gray", font=("TkDefaultFont", 8),
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Button(f, text="View Intensity Profile…", command=self._show_profile
                   ).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(8, 0))

    def _build_export_section(self, parent):
        f = ttk.LabelFrame(parent, text="Export", padding=8)
        f.grid(row=3, column=0, sticky="ew", padx=8, pady=4)
        f.columnconfigure(0, weight=1)
        ttk.Button(f, text="Export Results…", command=self.cmd_export
                   ).grid(row=0, column=0, sticky="ew")

    def _build_batch_section(self, parent):
        f = ttk.LabelFrame(parent, text="Batch Processing", padding=8)
        f.grid(row=4, column=0, sticky="ew", padx=8, pady=4)
        f.columnconfigure(0, weight=1)
        f.columnconfigure(1, weight=1)

        self.folder_label = ttk.Label(f, text="No folder selected",
                                      foreground="gray", wraplength=240, justify=tk.LEFT)
        self.folder_label.grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Button(f, text="Select Folder…", command=self.cmd_load_folder
                   ).grid(row=1, column=0, sticky="ew", padx=(0, 2), pady=(6, 0))
        self.batch_btn = ttk.Button(f, text="Run Batch", command=self.cmd_batch)
        self.batch_btn.grid(row=1, column=1, sticky="ew", padx=(2, 0), pady=(6, 0))

        self.batch_progress = ttk.Progressbar(f, mode="determinate")
        self.batch_progress.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        self.batch_status = ttk.Label(f, text="", foreground="gray",
                                      font=("TkDefaultFont", 8), wraplength=240)
        self.batch_status.grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

    # ── Commands ──────────────────────────────────────────────────────────────

    def cmd_load_image(self):
        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.tiff *.tif *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        self.image_path = Path(path)
        self._set_status(f"Loading {self.image_path.name}…")
        try:
            self.orig_image = ip.load_image(path)
            self.gray_image = ip.to_grayscale(self.orig_image)
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            return

        # Clear previous results
        self.mask = self.contour = self.centerline = None
        self.heatmap = self.cl_profile = None
        self.body_intensity = self.cl_intensity = None
        self.body_int_var.set("—")
        self.cl_int_var.set("—")

        self.file_label.config(text=self.image_path.name, foreground="black")
        self.view_var.set("Original")
        self._refresh_canvas()
        h, w = self.orig_image.shape[:2]
        self._set_status(f"Loaded: {self.image_path.name}  ({w} × {h} px)")

    def cmd_load_folder(self):
        path = filedialog.askdirectory(title="Select Image Folder")
        if not path:
            return
        self.folder_path = Path(path)
        n = sum(1 for f in self.folder_path.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        self.folder_label.config(
            text=f"{self.folder_path.name}/  ({n} images)", foreground="black"
        )
        self._set_status(f"Folder: {self.folder_path}  ({n} images found)")

    def cmd_process(self):
        if self.orig_image is None:
            messagebox.showinfo("No Image", "Please open an image first.")
            return

        self._set_status("Processing…")
        self.root.config(cursor="watch")
        self.root.update()

        try:
            thresh_raw = self.thresh_var.get()
            threshold = None if thresh_raw == 0 else thresh_raw

            self.mask      = ip.detect_mask(self.orig_image, threshold)
            self.contour   = ip.get_contour(self.mask)
            self.centerline = ip.compute_centerline(self.mask, self.smooth_var.get())
            self.heatmap   = ip.create_heatmap(self.gray_image, self.mask, self.cmap_var.get())
            self.body_intensity = ip.compute_body_intensity(self.gray_image, self.mask)
            self.cl_intensity, self.cl_profile = ip.compute_centerline_intensity(
                self.gray_image, self.centerline
            )

            self.body_int_var.set(f"{self.body_intensity:.2f}")
            self.cl_int_var.set(f"{self.cl_intensity:.2f}")
            self.view_var.set("Heatmap + Overlay")
            self._refresh_canvas()
            self._set_status(
                f"Done — body mean: {self.body_intensity:.2f} | "
                f"centerline mean: {self.cl_intensity:.2f}  (lower = darker = more melanized)"
            )
        except Exception as e:
            messagebox.showerror("Processing Error", str(e))
            self._set_status("Processing failed.")
        finally:
            self.root.config(cursor="")

    def cmd_export(self):
        if self.mask is None:
            messagebox.showinfo("Nothing to Export", "Process an image first.")
            return

        out_dir = filedialog.askdirectory(title="Select Output Folder")
        if not out_dir:
            return

        try:
            overlay = ip.draw_overlay(self.heatmap, self.contour, self.centerline)
            out = ip.export_single(
                out_dir,
                self.image_path.stem if self.image_path else "result",
                self.body_intensity, self.cl_intensity,
                self.cl_profile, overlay,
            )
            messagebox.showinfo("Exported", f"Results saved to:\n{out}")
            self._set_status(f"Exported to {out}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))

    def cmd_batch(self):
        if self.folder_path is None:
            messagebox.showinfo("No Folder", "Select a folder first.")
            return

        paths = sorted(
            p for p in self.folder_path.iterdir() if p.suffix.lower() in IMAGE_EXTS
        )
        if not paths:
            messagebox.showinfo("No Images", "No supported images found in that folder.")
            return

        out_dir = filedialog.askdirectory(title="Select Output Folder")
        if not out_dir:
            return

        thresh_raw = self.thresh_var.get()
        threshold  = None if thresh_raw == 0 else thresh_raw
        colormap   = self.cmap_var.get()

        self.batch_progress.configure(maximum=len(paths), value=0)
        self.batch_btn.configure(state="disabled")

        def _run():
            import pandas as pd

            results = []
            heatmaps = []

            for i, path in enumerate(paths):
                self.root.after(0, lambda i=i, n=path.name: (
                    self.batch_status.config(text=f"{i+1}/{len(paths)}: {n}"),
                    self.batch_progress.configure(value=i + 1),
                ))
                try:
                    img   = ip.load_image(str(path))
                    gray  = ip.to_grayscale(img)
                    mask  = ip.detect_mask(img, threshold)
                    cl    = ip.compute_centerline(mask, self.smooth_var.get())
                    b_int = ip.compute_body_intensity(gray, mask)
                    c_int, c_prof = ip.compute_centerline_intensity(gray, cl)
                    hmap  = ip.create_heatmap(gray, mask, colormap)
                    overlay = ip.draw_overlay(hmap, ip.get_contour(mask), cl)

                    ip.export_single(out_dir, path.stem, b_int, c_int, c_prof, overlay)

                    results.append({
                        "image": path.name,
                        "whole_body_mean_intensity": round(b_int, 4),
                        "dorsal_centerline_mean_intensity": round(c_int, 4),
                    })
                    heatmaps.append(hmap.astype(float))

                except Exception as e:
                    results.append({"image": path.name, "error": str(e)})

            # Summary CSV
            out = Path(out_dir)
            pd.DataFrame(results).to_csv(out / "batch_summary.csv", index=False)

            # Average heatmap
            avg = None
            if heatmaps:
                target = heatmaps[0].shape[:2]
                resized = [
                    cv2.resize(h.astype(np.uint8), (target[1], target[0]))
                    if h.shape[:2] != target else h.astype(np.uint8)
                    for h in heatmaps
                ]
                avg = np.mean(resized, axis=0).astype(np.uint8)
                cv2.imwrite(
                    str(out / "average_heatmap.png"),
                    cv2.cvtColor(avg, cv2.COLOR_RGB2BGR),
                )

            self.root.after(0, lambda: (
                self.batch_btn.configure(state="normal"),
                self.batch_status.config(text=f"Done — {len(results)} images processed."),
                self._set_status(f"Batch complete. Results in {out_dir}"),
                messagebox.showinfo(
                    "Batch Complete",
                    f"Processed {len(results)} images.\nResults saved to:\n{out_dir}",
                ),
                (self._show_heatmap_window(avg, "Average Heatmap Across Batch") if avg is not None else None),
            ))

        threading.Thread(target=_run, daemon=True).start()

    # ── Display ───────────────────────────────────────────────────────────────

    def _refresh_canvas(self):
        self.ax.clear()
        self.ax.axis("off")

        view = self.view_var.get()

        if view == "Original" and self.orig_image is not None:
            rgb = cv2.cvtColor(self.orig_image, cv2.COLOR_BGR2RGB)
            self.ax.imshow(rgb)
            self.ax.set_title("Original", fontsize=10)

        elif view == "Grayscale" and self.gray_image is not None:
            self.ax.imshow(self.gray_image, cmap="gray", vmin=0, vmax=255)
            self.ax.set_title("Grayscale", fontsize=10)

        elif view == "Heatmap" and self.heatmap is not None:
            self.ax.imshow(self.heatmap)
            self.ax.set_title(f"Heatmap  ({self.cmap_var.get()})", fontsize=10)

        elif view == "Heatmap + Overlay" and self.heatmap is not None:
            overlay = ip.draw_overlay(self.heatmap, self.contour, self.centerline)
            self.ax.imshow(overlay)
            self.ax.set_title("Heatmap + Contour + Centerline", fontsize=10)

        elif self.orig_image is not None:
            # Fallback if processed view selected before processing
            self.ax.imshow(cv2.cvtColor(self.orig_image, cv2.COLOR_BGR2RGB))
            self.ax.set_title("Original", fontsize=10)

        else:
            self.ax.set_facecolor("#e0e0e0")
            self.ax.text(0.5, 0.5, "Open an image to begin",
                         transform=self.ax.transAxes,
                         ha="center", va="center", fontsize=13, color="#888888")

        self.fig.tight_layout(pad=0.3)
        self.canvas.draw()

    def _refresh_heatmap(self):
        if self.mask is None:
            return
        self.heatmap = ip.create_heatmap(self.gray_image, self.mask, self.cmap_var.get())
        self._refresh_canvas()

    # ── Pop-up windows ────────────────────────────────────────────────────────

    def _show_profile(self):
        if self.cl_profile is None or len(self.cl_profile) == 0:
            messagebox.showinfo("No Data", "Process an image first.")
            return

        win = tk.Toplevel(self.root)
        win.title("Dorsal Centerline Intensity Profile")
        win.geometry("620x420")

        fig = Figure(figsize=(6, 4), dpi=96)
        ax = fig.add_subplot(111)
        x = np.arange(len(self.cl_profile))
        ax.plot(x, self.cl_profile, color="#1f77b4", linewidth=1.5)
        ax.fill_between(x, self.cl_profile, alpha=0.15, color="#1f77b4")
        ax.set_xlabel("Position along centerline (px)")
        ax.set_ylabel("Mean pixel intensity  (lower = darker)")
        ax.set_title("Dorsal Centerline Intensity Profile")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        c = FigureCanvasTkAgg(fig, master=win)
        c.draw()
        c.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(c, win).update()

    def _show_heatmap_window(self, rgb_image: np.ndarray, title: str):
        win = tk.Toplevel(self.root)
        win.title(title)
        win.geometry("720x560")

        fig = Figure(figsize=(7, 5), dpi=96)
        ax = fig.add_subplot(111)
        ax.imshow(rgb_image)
        ax.axis("off")
        ax.set_title(title, fontsize=11)
        fig.tight_layout()

        c = FigureCanvasTkAgg(fig, master=win)
        c.draw()
        c.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(c, win).update()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _on_thresh_change(self, value):
        v = int(float(value))
        self.thresh_label.config(text="Auto (Otsu)" if v == 0 else str(v))

    def _set_status(self, msg: str):
        self.status_var.set(msg)

    def _show_about(self):
        messagebox.showinfo(
            "About — Cuticle Melanization Analyzer",
            "Cuticle Melanization Analyzer\n\n"
            "Analyzes caterpillar cuticle melanization from overhead images\n"
            "on a white background.\n\n"
            "Outputs per image:\n"
            "  • Whole-body mean pixel intensity\n"
            "  • Dorsal centerline mean intensity & profile\n"
            "  • Colorized heatmap PNG\n"
            "  • CSV summary and profile files\n\n"
            "Heatmap convention:\n"
            "  Brighter/hotter color = darker in original = more melanized.\n"
            "  Numeric intensity: lower value = darker = more melanized.\n\n"
            "Threshold slider:\n"
            "  0 = auto (Otsu) | 1–254 = manual level\n\n"
            "Supported formats: JPG, PNG, TIFF, BMP",
        )


def main():
    root = tk.Tk()

    # Improve DPI rendering on HiDPI/Retina displays
    try:
        root.tk.call("tk", "scaling", 1.4)
    except Exception:
        pass

    # Windows: set DPI awareness
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

    MelanizationApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
