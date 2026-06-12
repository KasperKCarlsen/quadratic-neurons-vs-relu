"""Draw a block-structure diagram for each model configuration.

Schematic only - no GPU and no model instantiation. The architecture is fully
determined by (model, ReLU, num_blocks), which we read from the study DBs in
optuna_databases/ (or you can pass them explicitly). One PNG per config is
written to Visualization/<config>_architecture.png.

This is drawn by hand on purpose: generic tools (torchinfo / torchviz) would
just list Conv2d layers and hide the convr*convg + convb*x^2 quadratic-neuron
structure, which is the whole point of the comparison.

Usage:
    python visualize_architecture.py
    python visualize_architecture.py --model Quad_neuron --ReLU 2 --num_blocks 5 5 5
"""

import os
import ast
import glob
import argparse

import matplotlib
matplotlib.use("Agg")  # headless: no display needed
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

QUAD_MODELS = {"Quad_neuron", "Quad_adapters"}  # the rest use plain conv blocks


def parse_study_db(db_path):
    """(model, relu, num_blocks) from an optuna_databases/*_study.db filename."""
    name_base = os.path.basename(db_path)[: -len("_study.db")]
    model, rest = name_base.split("_with_ReLU", 1)
    relu_str, arc_str = rest.split("_arc_", 1)
    return model, int(relu_str), ast.literal_eval(arc_str)


def _box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                linewidth=1.5, edgecolor="black", facecolor=color))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8)


def block_lines(model, relu):
    """The per-block forward, as readable pseudo-equations, adapted to model
    type and ReLU mode. Mirrors the actual BasicBlock.forward implementations."""
    if model in QUAD_MODELS:
        relu1 = "ReLU" if relu == 2 else "(no ReLU)"
        relu2 = "ReLU" if relu in (1, 2) else "(no ReLU)"
        return [
            f"Quadratic block   (ReLU mode = {relu})",
            "",
            "  z1 = BN( conv_r(x) * conv_g(x) + conv_b(x^2) )",
            f"  z1 = {relu1}( z1 )",
            "  z2 = BN( conv_r(z1) * conv_g(z1) + conv_b(z1^2) )",
            "  out = z2 + shortcut(x)",
            f"  out = {relu2}( out )",
            "",
            "  conv_g init: w=0, b=1  ->  block starts LINEAR",
            "  conv_b init: w=0, b=0  ->  square term starts OFF",
        ]
    return [
        "Baseline (linear) block",
        "",
        "  z1 = ReLU( BN( conv(x) ) )",
        "  z2 = BN( conv(z1) )",
        "  out = ReLU( z2 + shortcut(x) )",
        "",
        "  (ReLUs always on; --ReLU flag does not alter this block)",
    ]


def draw_architecture(model, relu, num_blocks, classes, out_path):
    is_quad = model in QUAD_MODELS
    stage_color = "#ffe0b2" if is_quad else "#c8e6c9"  # orange (quad) / green (linear)
    stem_relu = "+ReLU" if (not is_quad or relu != 0) else "(no ReLU)"

    # Macro pipeline: (label, fill colour).
    boxes = [
        ("Input\n3 x 32 x 32", "#eeeeee"),
        (f"Conv 3x3\n3 -> 16\nBN {stem_relu}", "#bbdefb"),
        (f"Stage 1\nx{num_blocks[0]}\n16 ch\n32 x 32", stage_color),
        (f"Stage 2\nx{num_blocks[1]}\n32 ch  (/2)\n16 x 16", stage_color),
        (f"Stage 3\nx{num_blocks[2]}\n64 ch  (/2)\n8 x 8", stage_color),
        ("Global\nAvgPool\n-> 64", "#eeeeee"),
        (f"Linear\n64 -> {classes}", "#bbdefb"),
    ]
    n = len(boxes)

    fig = plt.figure(figsize=(max(11, n * 1.7), 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.25)

    # --- top: macro pipeline ---
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlim(0, n)
    ax1.set_ylim(0, 1)
    ax1.axis("off")
    ax1.set_title(f"{model}   ReLU={relu}   num_blocks={num_blocks}",
                  fontsize=12, fontweight="bold")
    for i, (label, color) in enumerate(boxes):
        _box(ax1, i + 0.08, 0.28, 0.84, 0.46, label, color)
        if i < n - 1:
            ax1.annotate("", xy=(i + 1.06, 0.51), xytext=(i + 0.94, 0.51),
                         arrowprops=dict(arrowstyle="-|>", lw=1.3, color="black"))

    # --- bottom: one block's internal structure ---
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    lines = block_lines(model, relu)
    ax2.text(0.5, 0.93, "Inside one block", ha="center", va="top",
             fontsize=11, fontweight="bold", transform=ax2.transAxes)
    ax2.text(0.06, 0.80, "\n".join(lines), ha="left", va="top",
             fontsize=10, family="monospace", transform=ax2.transAxes,
             bbox=dict(boxstyle="round,pad=0.6", facecolor=stage_color,
                       edgecolor="black", alpha=0.5))

    os.makedirs("Visualization", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Draw model block-structure diagrams.")
    parser.add_argument("--model", default=None, help="If set, draw only this model.")
    parser.add_argument("--ReLU", type=int, default=None, choices=[0, 1, 2])
    parser.add_argument("--num_blocks", type=int, nargs="+", default=None)
    parser.add_argument("--classes", type=int, default=10,
                        help="Number of output classes (for the head label).")
    args = parser.parse_args()

    if args.model and args.ReLU is not None and args.num_blocks:
        configs = [(args.model, args.ReLU, args.num_blocks)]
    else:
        configs = [parse_study_db(p)
                   for p in sorted(glob.glob("optuna_databases/*_study.db"))]
        print(f"Found {len(configs)} config(s) in optuna_databases/.")

    for model, relu, num_blocks in configs:
        config_name = f"{model}_ReLU{relu}_arc_{num_blocks}"
        draw_architecture(model, relu, num_blocks, args.classes,
                          f"Visualization/{config_name}_architecture.png")


if __name__ == "__main__":
    main()
