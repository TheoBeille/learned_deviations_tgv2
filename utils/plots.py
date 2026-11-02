import os
import shutil

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# matplotlib's default (tab10) colors
PAPER = {
    "blue":    "#1F77B4",
    "orange":  "#FF7F0E",
    "green":   "#2CA02C",
    "red":     "#D62728",
    "purple":  "#9467BD",
    "teal":    "#17BECF",
    "ochre":   "#BCBD22",
    "gray":    "#7F7F7F",
}


def _latex_available():
    """True if a LaTeX install can render a trivial figure."""
    if shutil.which("latex") is None:
        return False
    try:
        with matplotlib.rc_context({"text.usetex": True}):
            fig = plt.figure()
            fig.text(0.5, 0.5, r"$x^2$")
            fig.canvas.draw()
        return True
    except Exception:
        return False
    finally:
        plt.close("all")


def apply_paper_style(use_tex=True):
    """Paper-like matplotlib style. Text is rendered with LaTeX when it is
    installed, otherwise with matplotlib's mathtext."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.color": "#BBBBBB",
        "grid.linestyle": ":",
        "grid.alpha": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "lines.linewidth": 2.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "text.usetex": use_tex and _latex_available(),
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
    })


def _ensure_plots_dir(dir_name="plots"):
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def _save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(_ensure_plots_dir(), f"{name}.pdf"))
    plt.close()


# one fixed style per method, so that all figures are consistent: the color is the
# method (blue zero, green random, orange learned with safeguard, red learned without),
# the line style the variant (solid: read d, trained at 512; dotted: read x;
# dashed: trained at 128)
_ZERO, _ZERO_X = dict(color=PAPER["blue"], ls="-"), dict(color=PAPER["blue"], ls=":")
_SAFE, _SAFE_X = dict(color=PAPER["orange"], ls="-"), dict(color=PAPER["orange"], ls=":")
_NOSAFE = dict(color=PAPER["red"], ls="-")
STYLES = {
    "Zero deviation":                       _ZERO,
    "Zero deviation, read $d$":             _ZERO,
    "Zero deviation, read $x$":             _ZERO_X,
    "Random":                               dict(color=PAPER["green"], ls=":"),
    "Learned":                              _SAFE,
    "Learned, read $d$":                    _SAFE,
    "Learned $d$, safeguard":               _SAFE,
    "Learned (512 training)":               _SAFE,
    "Learned, read $x$":                    _SAFE_X,
    "Learned $x$, safeguard":               _SAFE_X,
    "Learned (no safeguard)":               _NOSAFE,
    "Learned $d$, no safeguard":            _NOSAFE,
    "Learned (512 training, no safeguard)": _NOSAFE,
    "Learned (128 training)":               dict(color=PAPER["orange"], ls="--"),
    "Learned (128 training, no safeguard)": dict(color=PAPER["red"], ls="--"),
    "PDHG":                                 dict(color=PAPER["purple"], ls="-"),
}


def plot_curves(curves, ylabel, name, loglog=True, slopes=False, horizon=None,
                hline=None, show=False, mark_first=True, iteration_start=1):
    """
    Plot one curve per method against the iteration number.

    curves  : dict {label: values, one per iteration}. Values of shape
              (n_slices, n_iterations) are drawn as their mean over the slices
              (geometric mean on a log scale); the spread over the slices is
              reported in the tables, not drawn
    slopes  : add O(1/n) and O(1/n^2) reference lines
    horizon : draw a vertical line at the training horizon
    hline   : (value, label) horizontal reference line
    mark_first : highlight the first recorded iterate at n=1
    iteration_start : iteration number assigned to the first value
    Saved as plots/<name>.pdf.
    """
    plt.figure(figsize=(6, 6))
    plot = plt.loglog if loglog else plt.plot
    curves = dict(curves)
    for label, values in list(curves.items()):
        values = np.asarray(values)
        if values.ndim == 2:
            curves[label] = np.exp(np.log(values).mean(0)) if loglog else values.mean(0)
    def visible(values):
        """(iterations, values), without n = 0 on a log scale."""
        values = np.asarray(values)
        iterations = np.arange(iteration_start, iteration_start + len(values))
        keep = iterations > 0 if loglog else slice(None)
        return iterations[keep], values[keep]

    for label, values in curves.items():
        style = STYLES.get(label, {})
        iterations, values = visible(values)
        plot(iterations, values, label=label, **style)
        if mark_first and len(values):
            plt.scatter([iterations[0]], [values[0]], color=style.get("color"),
                        s=28, zorder=3)

    if slopes:
        # anchored at the first plotted point of the first curve: c / n^p goes through it
        n, first = visible(next(iter(curves.values())))
        n = n[n > 0]
        c, n0 = first[-len(n)], n[0]
        plot(n, c * n0 / n, color=PAPER["gray"], lw=1.0, ls="--", label=r"$O(1/n)$")
        plot(n, c * n0 ** 2 / n ** 2, color=PAPER["gray"], lw=1.0, ls=":", label=r"$O(1/n^2)$")
    if horizon is not None:
        plt.axvline(horizon, color=PAPER["gray"], lw=1.0, ls="-.", label=f"$T={horizon}$ (training)")
    if hline is not None:
        plt.axhline(hline[0], color=PAPER["gray"], lw=1.0, ls="--", label=hline[1])

    if not loglog:
        plt.xlim(left=0)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.xlabel("iteration $n$")
    plt.ylabel(ylabel)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(_ensure_plots_dir(), f"{name}.pdf"))
    plt.show() if show else plt.close()


def show_images(images, name, clim=(0, 1), show=False):
    """
    Row of grayscale images. images : dict {title: tensor [1, 1, H, W]}.
    clim : gray-level range, or None to scale each image separately.
    Saved as plots/<name>.pdf.
    """
    vmin, vmax = clim if clim is not None else (None, None)
    fig, axes = plt.subplots(1, len(images), figsize=(3.2 * len(images), 3.6))
    for ax, (title, image) in zip(np.atleast_1d(axes), images.items()):
        ax.imshow(image.squeeze().cpu().numpy(), cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(_ensure_plots_dir(), f"{name}.pdf"))
    plt.show() if show else plt.close()
