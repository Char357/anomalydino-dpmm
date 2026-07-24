"""
Shared dark-theme plot style for the presentation figures, tuned to the dark-purple
slides (Impact headings, like the deck's Bebas Neue). The palette was validated with
the dataviz skill's computable checks: CVD-safe and high-contrast on the dark surface.
Bright/pastel marks are intentionally above the standard dark lightness band so they
pop on the deep-purple background (contrast is what matters on dark, and it passes).

Usage: at the top of a plotting script,
    from plot_style import use_style, BLUE, TEAL, GOLD, CORAL, PURPLE_RAMP, MUTED
    use_style()
then colour marks with the named constants (categorical in the given order).
"""

import matplotlib.pyplot as plt

# Surfaces / ink
SURFACE = "#160e30"     # figure + axes background (dark deep purple ~ slide)
INK     = "#ece8f7"     # primary text
MUTED   = "#b3a9d0"     # secondary text, axis ticks, reference lines
GRID    = "#322a52"     # subtle gridlines

# Categorical palette — bright pastels, CVD-safe & high-contrast on SURFACE; fixed order.
BLUE  = "#74a9f2"
TEAL  = "#35cbb0"
GOLD  = "#e2ba55"
CORAL = "#f090ab"
CATEGORICAL = [BLUE, TEAL, GOLD, CORAL]

# Sequential purple ramp for K-ordered series (fewest -> most components).
PURPLE_RAMP = ["#6f4fc0", "#8f72d4", "#af96e6", "#cfbdf2"]

# Heading font (closest installed match to the deck's Bebas Neue).
TITLE_FONT = "Impact"


def use_style():
    """Apply the dark theme globally. Call once before plotting."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.edgecolor": SURFACE,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "grid.alpha": 0.6,
        "legend.facecolor": SURFACE,
        "legend.edgecolor": MUTED,
        "legend.framealpha": 0.85,
        "axes.prop_cycle": plt.cycler(color=CATEGORICAL),
        # bigger + bolder throughout
        "font.family": TITLE_FONT,
        "font.size": 14,
        "axes.titlesize": 18,
        "axes.titleweight": "bold",
        "axes.labelsize": 15,
        "axes.labelweight": "bold",
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 13,
        "lines.linewidth": 2.6,
        "lines.markersize": 9,
        "axes.linewidth": 1.3,
        "figure.dpi": 150,
    })
