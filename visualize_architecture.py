"""Schematic diagrams of the QResNet models that make ReLU placement obvious.

The three model families (Baseline / Quad_neuron, and the adapter variant) all
share the same skeleton:

    stem(conv+bn) -> stage1 -> stage2 -> stage3 -> avgpool -> fc

What differs between runs is the `relu` level, which decides *where* the
rectifiers sit. This script mirrors the exact forward passes in
py_files/Baseline.py and py_files/Quad_Neuron.py:

  * QResNet.forward applies a ReLU after the stem and after each stage
    whenever relu != 0   (the "macro" ReLUs).
  * BasicBlock.forward applies a ReLU after conv1 and after the residual add
    whenever relu == 2   (the "internal" ReLUs).

So:
    relu 0 -> no ReLU anywhere (pure polynomial / linear skeleton)
    relu 1 -> macro ReLUs only (once per stage + after the stem)
    relu 2 -> macro ReLUs + two ReLUs inside every block

For each (model, relu level) we draw two panels:
    top    - the macro pipeline with ReLU badges between stages
    bottom - a single BasicBlock expanded, with internal ReLU badges

ReLUs are always drawn as bright red badges so their location is unmistakable.

Run:  python visualize_architecture.py            # default arch (3,3,3)
      python visualize_architecture.py 5 5 5      # custom arch
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless: no display needed on the cluster/CI
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT_DIR = "Visualization"

# ---- palette -------------------------------------------------------------
C_STEM = "#bcd4e6"      # stem / pooling / fc - neutral blue-grey
C_CONV_LIN = "#cfe8cf"  # linear conv (baseline)
C_CONV_QUAD = "#f6d8a8"  # quadratic conv (quad models)
C_BN = "#e8e8e8"        # batchnorm
C_ADD = "#d9c2e8"       # residual addition
C_RELU = "#e8443a"      # ReLU - deliberately loud
C_EDGE = "#444444"


def _box(ax, x, y, w, h, text, color, fontsize=9, bold=False):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.2, edgecolor=C_EDGE, facecolor=color,
        )
    )
    ax.text(
        x + w / 2, y + h / 2, text,
        ha="center", va="center", fontsize=fontsize,
        fontweight="bold" if bold else "normal", zorder=5,
    )


def _relu_badge(ax, x, y, w=0.7, h=0.42):
    """A loud red ReLU marker - the whole point of the figure."""
    _box(ax, x, y, w, h, "ReLU", C_RELU, fontsize=8.5, bold=True)
    # white text on red
    ax.texts[-1].set_color("white")


def _arrow(ax, x0, y0, x1, y1, style="-"):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>", mutation_scale=12,
            linewidth=1.2, color=C_EDGE, linestyle=style,
            shrinkA=0, shrinkB=0,
        )
    )


# ---- macro pipeline ------------------------------------------------------
def draw_macro_clean(ax, num_blocks, relu):
    """Top panel: Input -> Stem -> Stage1..3 -> Pool -> FC, with macro ReLUs."""
    macro_relu = relu != 0
    bw, bh = 1.9, 1.0
    pitch = bw + 1.15
    y = 0.9

    stages = [
        ("Input\n3x32x32", C_STEM, False),
        ("Stem\nconv3x3+BN", C_STEM, macro_relu),
        (f"Stage 1\n{num_blocks[0]} blk @16\n32x32", C_CONV_LIN, macro_relu),
        (f"Stage 2\n{num_blocks[1]} blk @32\n16x16", C_CONV_LIN, macro_relu),
        (f"Stage 3\n{num_blocks[2]} blk @64\n8x8", C_CONV_LIN, macro_relu),
        ("AvgPool\nflatten", C_STEM, False),
        ("FC\nclasses", C_STEM, False),
    ]

    for i, (label, color, has_relu) in enumerate(stages):
        x = 0.3 + i * pitch
        _box(ax, x, y, bw, bh, label, color, fontsize=8)
        if i > 0:
            x_prev = 0.3 + (i - 1) * pitch
            _arrow(ax, x_prev + bw, y + bh / 2, x, y + bh / 2)
            if stages[i - 1][2]:  # ReLU leaves the *previous* box
                bx = x_prev + bw + (pitch - bw) / 2 - 0.35
                _relu_badge(ax, bx, y + bh / 2 - 0.21)

    ax.set_xlim(0, 0.3 + len(stages) * pitch)
    ax.set_ylim(0, 2.4)
    ax.set_title(
        "Macro view  -  ReLU after stem & after each stage (active when relu != 0)",
        fontsize=10, loc="left", fontweight="bold",
    )
    ax.axis("off")


# ---- single block detail -------------------------------------------------
def draw_block(ax, relu, is_quad):
    """Bottom panel: one BasicBlock expanded with internal ReLUs (relu == 2)."""
    internal_relu = relu == 2

    conv_color = C_CONV_QUAD if is_quad else C_CONV_LIN
    if is_quad:
        conv1_lbl = "Quad conv1\n(r*g + b*x^2)"
        conv2_lbl = "Quad conv2\n(r*g + b*x^2)"
    else:
        conv1_lbl = "conv1 3x3\n(linear)"
        conv2_lbl = "conv2 3x3\n(linear)"

    bw, bh = 1.7, 0.85
    y = 1.6
    pitch = bw + 1.0

    # main path nodes (label, color); ReLU slots handled separately
    nodes = [
        ("x\n(input)", C_STEM),
        (conv1_lbl, conv_color),
        ("BN1", C_BN),
        # (optional ReLU here when relu==2)
        (conv2_lbl, conv_color),
        ("BN2", C_BN),
        ("(+)\nresidual add", C_ADD),
        # (optional ReLU here when relu==2)
        ("out", C_STEM),
    ]

    xs = [0.3 + i * pitch for i in range(len(nodes))]
    for (label, color), x in zip(nodes, xs):
        _box(ax, x, y, bw, bh, label, color, fontsize=8)

    # arrows along the main path
    for i in range(len(nodes) - 1):
        _arrow(ax, xs[i] + bw, y + bh / 2, xs[i + 1], y + bh / 2)

    # internal ReLU #1: after BN1 (index 2 -> 3)
    if internal_relu:
        bx = xs[2] + bw + (pitch - bw) / 2 - 0.35
        _relu_badge(ax, bx, y + bh / 2 - 0.21)
    # internal ReLU #2: after the residual add (index 5 -> 6)
    if internal_relu:
        bx = xs[5] + bw + (pitch - bw) / 2 - 0.35
        _relu_badge(ax, bx, y + bh / 2 - 0.21)

    # residual skip connection: from x (node 0) down and over to (+) (node 5)
    x_start = xs[0] + bw / 2
    x_end = xs[5] + bw / 2
    y_skip = y - 0.85
    _arrow(ax, x_start, y, x_start, y_skip)
    ax.plot([x_start, x_end], [y_skip, y_skip], color=C_EDGE, linewidth=1.2)
    _arrow(ax, x_end, y_skip, x_end, y)
    ax.text(
        (x_start + x_end) / 2, y_skip - 0.22,
        "shortcut / identity (residual stream - kept clean unless relu==2 clips it)",
        ha="center", va="center", fontsize=8, style="italic", color="#666",
    )

    ax.set_xlim(0, 0.3 + len(nodes) * pitch)
    ax.set_ylim(0, 3.2)
    ax.set_title(
        "Block detail  -  ReLU after conv1 & after residual add (active only when relu == 2)",
        fontsize=10, loc="left", fontweight="bold",
    )
    ax.axis("off")


# ---- relu counting (for the title) --------------------------------------
def count_relus(num_blocks, relu):
    if relu == 0:
        return 0
    macro = 1 + 3  # stem + 3 stages
    if relu == 1:
        return macro
    internal = 2 * sum(num_blocks)  # two per block
    return macro + internal


# ---- one full figure -----------------------------------------------------
def make_figure(model_name, is_quad, relu, num_blocks):
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(15, 8), gridspec_kw={"height_ratios": [1, 1.1]}
    )

    n = count_relus(num_blocks, relu)
    total_layers = 2 + 2 * sum(num_blocks)  # stem conv + 2 per block + fc
    fig.suptitle(
        f"{model_name}   |   relu = {relu}   |   arch {tuple(num_blocks)}  "
        f"({total_layers} weight layers)   |   {n} ReLU activations",
        fontsize=13, fontweight="bold",
    )

    draw_macro_clean(ax_top, num_blocks, relu)
    draw_block(ax_bot, relu, is_quad)

    # shared legend for the ReLU marker
    fig.text(
        0.5, 0.015,
        "Red 'ReLU' badge = a rectifier in the forward pass.   "
        "relu 0: none   |   relu 1: macro only (top panel)   |   relu 2: macro + internal (both panels)",
        ha="center", fontsize=9.5,
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    arc = "_".join(str(b) for b in num_blocks)
    out = f"{OUT_DIR}/ARCH_{model_name}_ReLU{relu}_({arc}).png"
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    if len(sys.argv) == 4:
        num_blocks = [int(a) for a in sys.argv[1:4]]
    else:
        num_blocks = [3, 3, 3]

    models = [
        ("Baseline", False),       # linear convs
        ("Quad_neuron", True),     # quadratic convs
    ]
    for model_name, is_quad in models:
        for relu in (0, 1, 2):
            make_figure(model_name, is_quad, relu, num_blocks)


if __name__ == "__main__":
    main()
