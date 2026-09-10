"""
Generates the three ICRA-paper figures (pipeline architecture, pheromone
reaction-primitive test setup, genotype/design-variable schema) as both
vector PDF (preferred for LaTeX/Overleaf - infinite zoom, tiny file size)
and 300 DPI PNG (fallback, e.g. for a quick preview or a venue that wants
raster). Run once from anywhere with the project's own matplotlib install:

    .venv/Scripts/python.exe paper/figures/generate_figures.py

No dependency on mujoco/torch/etc - pure matplotlib schematic drawing, so
this can be re-run any time the figures need a tweak without touching the
simulation code at all.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.linewidth": 0.8,
})

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
INK = "#1a1a1a"
ACCENT = "#2a5d9f"   # RL / learned components
ACCENT2 = "#b3541e"  # light / pheromone stimulus
GREY = "#6b6b6b"
BOXFACE = "#f2f2f0"


def _save(fig, name):
    pdf_path = os.path.join(OUT_DIR, f"{name}.pdf")
    png_path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", pdf_path)
    print("wrote", png_path)


def box(ax, xy, w, h, text, face=BOXFACE, edge=INK, fontsize=8, textcolor=INK, lw=1.0, style="round,pad=0.02,rounding_size=0.05"):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=lw,
                        edgecolor=edge, facecolor=face, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=textcolor, zorder=3, linespacing=1.3)
    return p


def arrow(ax, p0, p1, color=INK, lw=1.1, style="-|>", connectionstyle="arc3,rad=0.0", ls="-"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=10,
                         linewidth=lw, color=color, connectionstyle=connectionstyle,
                         linestyle=ls, zorder=1)
    ax.add_patch(a)
    return a


# ---------------------------------------------------------------------
# Figure 1: RL-guided NSGA-III evolutionary pipeline (wide, double-column)
# ---------------------------------------------------------------------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.16, 2.75))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")

    stages = [
        "Population\n$P_g$",
        "Mutation +\ncrossover",
        "Symmetry\nmirror",
        "MJCF +\ncollision gate",
        "$B$-sweep +\nbaseline gait",
        "Light-response\ntests",
        "Objectives\n$f_1,\\ldots,f_7$",
        "NSGA-III\nselection",
    ]
    n = len(stages)
    x0, gap, w, h = 1.5, 12.1, 9.7, 17
    y = 15
    xs = [x0 + i * gap for i in range(n)]

    for i, (x, s) in enumerate(zip(xs, stages)):
        face = BOXFACE
        if i == 5:
            face = "#f7e9dd"
        box(ax, (x, y), w, h, s, face=face, fontsize=7.0)
        if i < n - 1:
            arrow(ax, (x + w, y + h / 2), (xs[i + 1], y + h / 2))

    # generational feedback loop, routed as a clean under-pass well below the row
    loop_y = y - 8.5
    ax.plot([xs[-1] + w / 2, xs[-1] + w / 2], [y, loop_y], color=GREY, lw=1.0, linestyle=(0, (4, 2)))
    ax.plot([xs[-1] + w / 2, xs[0] + w / 2], [loop_y, loop_y], color=GREY, lw=1.0, linestyle=(0, (4, 2)))
    arrow(ax, (xs[0] + w / 2, loop_y), (xs[0] + w / 2, y), color=GREY, lw=1.0, ls=(0, (4, 2)))
    ax.text((xs[-1] + w / 2 + xs[0] + w / 2) / 2, loop_y - 3.2,
            "generation $g \\rightarrow g+1$", ha="center", fontsize=7, color=GREY)

    # RL policy sub-block, feeding the breeding stage and reading reward from
    # objectives - routed as a clean over-pass above the row so it never
    # crosses through any box's interior.
    rl_x, rl_y, rl_w, rl_h = xs[1] - 0.8, y + h + 9.5, 12.0, 10
    box(ax, (rl_x, rl_y), rl_w, rl_h, "PPO policy\n(Graph Transformer\nactor--critic)",
        face="#dce8f7", edge=ACCENT, textcolor=ACCENT, fontsize=6.8)

    action_x = xs[1] + w / 2
    arrow(ax, (rl_x + rl_w * 0.35, rl_y), (action_x, y + h), color=ACCENT,
          connectionstyle="arc3,rad=-0.1")
    ax.text(rl_x + rl_w * 0.35 - 2.2, rl_y - 3.6, "action", fontsize=6.3, color=ACCENT, ha="center")

    over_y = y + h + 4.0  # channel height: clears every box top, stays below the RL box
    reward_x = xs[6] + w / 2
    rl_in_x = rl_x + rl_w * 0.85
    ax.plot([reward_x, reward_x], [y + h, over_y], color=ACCENT, lw=1.0)
    ax.plot([reward_x, rl_in_x], [over_y, over_y], color=ACCENT, lw=1.0)
    arrow(ax, (rl_in_x, over_y), (rl_in_x, rl_y), color=ACCENT, lw=1.0)
    ax.text((reward_x + rl_in_x) / 2, over_y + 1.6,
            "reward $=\\Delta\\,\\mathrm{scalarize}(f)$", fontsize=6.3, color=ACCENT, ha="center")

    ax.text(0, 43, "(a)", fontsize=9, weight="bold", va="top")
    _save(fig, "fig_pipeline")


# ---------------------------------------------------------------------
# Figure 2: Pheromone reaction-primitive test setup (single column)
# ---------------------------------------------------------------------
def _robot_blob(ax, cx, cy, scale=1.0, sensors=()):
    """A simple top-down capsule-shaped robot footprint with small
    triangular sensor markers at given (dx, dy) offsets from center."""
    body = mpatches.FancyBboxPatch((cx - 0.85 * scale, cy - 1.3 * scale),
                                    1.7 * scale, 2.6 * scale,
                                    boxstyle="round,pad=0,rounding_size=0.55",
                                    linewidth=1.1, edgecolor=INK, facecolor="#e7e7e3", zorder=3)
    ax.add_patch(body)
    ax.plot([cx], [cy + 1.55 * scale], marker=(3, 0, 0), markersize=6, color=INK, zorder=4)  # heading tick ("front")
    for dx, dy in sensors:
        ax.plot([cx + dx * scale], [cy + dy * scale], marker="D", markersize=4.2,
                 color=ACCENT2, markeredgecolor=INK, markeredgewidth=0.4, zorder=5)


def fig_reaction_primitives():
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.05))

    panels = [
        dict(ax=axes[0], title="(a) Left stage $\\rightarrow f_6$",
             stim_xy=(-2.6, 0), stim_wh=(2.6, 5.2),
             sensors=[(-0.55, 0.7), (-0.55, -0.3)],
             arrow=("yaw", "$\\Delta\\psi_L-\\Delta\\psi_b$")),
        dict(ax=axes[1], title="(b) Front stage $\\rightarrow f_7$",
             stim_xy=(-2.6, 0.9), stim_wh=(5.2, 2.6),
             sensors=[(-0.55, 0.7), (0.55, 0.7)],
             arrow=("speed", "$(v_F-v_b)/v_b$")),
    ]

    for p in panels:
        ax = p["ax"]
        ax.set_xlim(-3.1, 3.1)
        ax.set_ylim(-3.65, 4.0)
        ax.set_aspect("equal")
        ax.axis("off")

        # arena
        ax.add_patch(Rectangle((-2.9, -2.9), 5.8, 5.8, fill=False,
                                edgecolor=GREY, linewidth=0.7, linestyle=(0, (3, 2))))
        # ceiling light stimulus footprint
        sx, sy = p["stim_xy"]
        sw, sh = p["stim_wh"]
        ax.add_patch(Rectangle((sx, sy - 2.9), sw, sh, facecolor="#f7dfba",
                                edgecolor=ACCENT2, linewidth=1.0, alpha=0.9, zorder=1))
        _robot_blob(ax, 0, -0.3, scale=0.62, sensors=p["sensors"])

        kind, label = p["arrow"]
        if kind == "yaw":
            arrow(ax, (0.95, 0.7), (-0.95, 0.7), lw=1.3,
                  connectionstyle="arc3,rad=0.55")
            ax.text(0, 2.15, label, ha="center", fontsize=7.5)
        else:
            arrow(ax, (0, 1.55), (0, 2.55), lw=1.5)
            ax.text(0, 2.85, label, ha="center", fontsize=7.5)

        ax.text(0, -3.5, p["title"], ha="center", va="top", fontsize=7.8)

    fig.subplots_adjust(wspace=0.12)
    # shared legend
    handles = [
        mpatches.Patch(facecolor="#f7dfba", edgecolor=ACCENT2, label="light stimulus footprint"),
        plt.Line2D([0], [0], marker="D", color="none", markerfacecolor=ACCENT2,
                   markeredgecolor=INK, markersize=5, label="light-sensitive joint"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=1, fontsize=6.6,
               frameon=False, bbox_to_anchor=(0.5, -0.09))
    _save(fig, "fig_reaction_primitives")


# ---------------------------------------------------------------------
# Figure 3: Genotype / design-variable schema (single column)
# ---------------------------------------------------------------------
def fig_genotype_schema():
    fig = plt.figure(figsize=(3.45, 2.85))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.35)
    axL = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])

    # ---- Left: half-graph + auto-mirrored full graph ----
    axL.set_xlim(-1.6, 1.9)
    axL.set_ylim(-0.3, 4.6)
    axL.set_aspect("equal")
    axL.set_anchor("N")  # top-align within its gridspec cell, so (a)/(b) captions line up despite differing y-ranges
    axL.axis("off")
    axL.text(0.15, 4.45, "(a) Genotype graph", ha="center", fontsize=8)

    root = (0, 4.0)
    axL.add_patch(Circle(root, 0.28, facecolor="#dce8f7", edgecolor=ACCENT, linewidth=1.2, zorder=3))
    axL.text(*root, "$m_1$", ha="center", va="center", fontsize=7.5, zorder=4)

    spine = (0, 2.7)
    axL.add_patch(Circle(spine, 0.24, facecolor="#dce8f7", edgecolor=ACCENT, linewidth=1.2, zorder=3))
    arrow(ax=axL, p0=(root[0], root[1] - 0.28), p1=(spine[0], spine[1] + 0.24), lw=1.0)
    axL.text(0.55, 3.35, "port 1\n(spine)", fontsize=5.8, color=GREY, ha="left")

    evo_nodes = [(-1.05, 1.55), (-1.35, 0.55)]
    prev = spine
    for i, n_xy in enumerate(evo_nodes):
        axL.add_patch(Circle(n_xy, 0.22, facecolor="#e2f2df", edgecolor="#3f8f3f", linewidth=1.1, zorder=3))
        arrow(ax=axL, p0=prev, p1=n_xy, lw=0.9)
        prev = n_xy
    axL.text(-1.55, 1.85, "port 2\n(evolvable)", fontsize=5.8, color="#3f8f3f", ha="left")

    mir_nodes = [(1.05, 1.55), (1.35, 0.55)]
    prev = spine
    for n_xy in mir_nodes:
        axL.add_patch(Circle(n_xy, 0.22, facecolor="#eee", edgecolor=GREY, linewidth=1.0, linestyle=(0, (2, 1.5)), zorder=3))
        arrow(ax=axL, p0=prev, p1=n_xy, lw=0.9, color=GREY, style="-|>", connectionstyle="arc3,rad=0.0")
        prev = n_xy
    axL.text(1.05, 1.85, "port 3\n(auto-mirrored)", fontsize=5.8, color=GREY, ha="left")

    for n_xy in evo_nodes + mir_nodes:
        axL.plot(n_xy[0], n_xy[1], marker="D", markersize=3.6, color=ACCENT2,
                  markeredgecolor=INK, markeredgewidth=0.3, zorder=5)

    # ---- Right: one foldable joint's design variables ----
    axR.set_xlim(-1.7, 1.7)
    axR.set_ylim(-1.1, 4.6)
    axR.set_aspect("equal")
    axR.set_anchor("N")
    axR.axis("off")
    axR.text(0, 4.45, "(b) Per-joint variables", ha="center", fontsize=8)

    cy = 2.3
    box(axR, (-1.55, cy - 0.55), 3.1, 1.1, "foldable joint", face="#e2f2df",
        edge="#3f8f3f", fontsize=7.2)

    # angle dial
    dial_c = (0, cy - 1.55)
    axR.add_patch(mpatches.Wedge(dial_c, 0.85, 0, 45, facecolor="#f2f2f0", edgecolor=INK, linewidth=0.9))
    axR.plot([dial_c[0], dial_c[0] + 0.85], [dial_c[1], dial_c[1]], color=INK, lw=0.9)
    ang0 = np.radians(18)
    axR.plot([dial_c[0], dial_c[0] + 0.85 * np.cos(ang0)],
              [dial_c[1], dial_c[1] + 0.85 * np.sin(ang0)], color="#3f8f3f", lw=1.4)
    axR.text(dial_c[0] + 0.5, dial_c[1] + 0.18, r"$\theta_i$", fontsize=7, color="#3f8f3f")
    ang1 = np.radians(40)
    axR.plot([dial_c[0], dial_c[0] + 0.85 * np.cos(ang1)],
              [dial_c[1], dial_c[1] + 0.85 * np.sin(ang1)], color=ACCENT2, lw=1.4, linestyle=(0, (3, 1.5)))
    axR.text(dial_c[0] + 0.55, dial_c[1] + 0.62, r"$\theta_{light}$", fontsize=7, color=ACCENT2)
    axR.text(dial_c[0], dial_c[1] - 1.05, "baseline vs.\nlight-triggered\nhinge angle,\n$[0^{\\circ},45^{\\circ}]$",
              ha="center", va="top", fontsize=6.3, linespacing=1.3)

    axR.plot([0], [cy + 0.9], marker="D", markersize=5, color=ACCENT2,
              markeredgecolor=INK, markeredgewidth=0.4, zorder=5)
    axR.text(0.3, cy + 0.9, "light-sensitive\nPVC strip\n(selection\nvariable)", fontsize=6.3, va="center", linespacing=1.2)
    arrow(ax=axR, p0=(0, cy + 0.75), p1=(0, cy + 0.55), lw=0.9)

    _save(fig, "fig_genotype_schema")


if __name__ == "__main__":
    fig_pipeline()
    fig_reaction_primitives()
    fig_genotype_schema()
    print("All figures generated in", OUT_DIR)
