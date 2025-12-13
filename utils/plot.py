import numpy as np
import matplotlib.pyplot as plt

def plot_r2(
    x_ls,
    r2_lb,
    r2_oos,
    lower_bound,
    inf_r2=None,
    confidence_level=95,
    x_label="z",
    caption=None,
    to_show=True,
):
    """
    Plot helper for the tutorial / paper-style figure:

      - r2_oos(z): realized out-of-sample R^2
      - r2_lb(z):  asymptotic LLG-adjusted lower bound
      - lower_bound(z): one-sided CLT-adjusted lower CI for population R^2_*

    Uses a log-x axis (semilogx), typical when x_ls is a ridge grid (z values).
    """
    x_ls = np.asarray(x_ls).ravel()
    r2_lb = np.asarray(r2_lb).ravel()
    r2_oos = np.asarray(r2_oos).ravel()
    lower_bound = np.asarray(lower_bound).ravel()

    # Set a clean white background + simple "paper-style" defaults
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "black",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "grid.color": "0.85",
        "grid.linestyle": "-",
        "grid.linewidth": 1.0,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 14,
        "axes.labelsize": 16,
        "legend.fontsize": 12,
    })

    # Colors: tab10 defaults (blue for LB, green for R^2_OOS, purple-ish for infeasible line)
    tab10 = plt.get_cmap("tab10").colors
    color_lb  = tab10[0]
    color_oos = tab10[2]
    color_inf = tab10[3]
    color_fill = (0.7, 0.75, 0.9, 0.4)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    # Optional horizontal reference line for infeasible population R^2_* (if known in simulation)
    if inf_r2 is not None:
        ax.axhline(y=float(inf_r2), color=color_inf, linestyle="--", linewidth=2.0, label=r"$R_{*}^2$")

    ax.semilogx(x_ls, r2_lb, color=color_lb, linewidth=2.2, label="Asymptotic LB")
    ax.semilogx(x_ls, r2_oos, color=color_oos, linewidth=2.2, label=r"$R^2_{OOS}$")

    # Lower CI curve (dashed)
    ax.semilogx(x_ls, lower_bound, color=color_lb, linewidth=1.8, linestyle="--",
                label=f"{confidence_level:g}% lower CI")

    ax.set_xlabel(x_label)
    ax.set_ylabel(r"$R^2$ (values)")

    # Shade the one-sided CI region up to 1.0, clipped if lower_bound exceeds 1
    y_top = 1.0
    lb_clip = np.clip(lower_bound, -np.inf, y_top)

    ax.fill_between(
        x_ls,
        lb_clip,
        np.ones_like(x_ls) * y_top,
        color=color_fill,
        label=f"{confidence_level:g}% CI area",
    )

    # Optional figure caption under the axis
    if caption is not None:
        ax.text(0.5, -0.15, caption, transform=ax.transAxes, ha="center", va="top", fontsize=10)

    ax.legend(loc="lower right", frameon=True, fancybox=True, framealpha=0.8)

    if to_show:
        plt.show()

    return fig, ax
