import contextlib
import io
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap



BLUE_FADE = LinearSegmentedColormap.from_list(
    "blue_fade", ["#1f4e79", "#5b88b5", "#a8c5e0"]
)
PURPLE_FADE = LinearSegmentedColormap.from_list(
    "purple_fade", ["#4b0082", "#8b5fbf", "#d4bff0"]
)
GREEN_FADE = LinearSegmentedColormap.from_list(
    "green_fade", ["#0b5e2a", "#4caf50", "#a5d6a7"]
)
ORANGE_FADE = LinearSegmentedColormap.from_list(
    "orange_fade", ["#cc4f00", "#ed8936", "#fbc78b"]
)



CURVE_LW = 3


HERE   = Path(__file__).parent
SCRIPT = HERE / "kaplow_becker_irs_3may_HPcal.py"

T_REFORM_RE = re.compile(r"^t_reform\s*=\s*np\.array\(\[[^\]]*\]\).*$", re.MULTILINE)
SANITY_RE   = re.compile(r"^S_g0\s*=\s*_compute_S_at_gamma\b.*$",       re.MULTILINE)


def _scalar_re(name: str) -> re.Pattern:
    """Match `name = ...` assignment at the start of a line."""
    return re.compile(rf"^{re.escape(name)}\s*=.*$", re.MULTILINE)


def sweep(t1_values: np.ndarray, overrides: dict[str, float], tag: str):
    """Run the IRS-HPcal script once per t1 value, with module-level scalars
    overridden according to `overrides`.  Returns (S array, G_base)."""
    src = SCRIPT.read_text()
    for name, val in overrides.items():
        src = _scalar_re(name).sub(f"{name} = {val!r}", src, count=1)

    src = SANITY_RE.sub("S_g0 = 0.0  # stubbed for sweep speed", src, count=1)

    out = np.full_like(t1_values, np.nan, dtype=float)
    G_base = None
    for k, t1 in enumerate(t1_values):
        patched = T_REFORM_RE.sub(
            f"t_reform = np.array([{t1:.10f}, 0.25, 0.25])",
            src, count=1,
        )
        ns = {"__name__": "__patched__", "__file__": str(SCRIPT)}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(patched, str(SCRIPT), "exec"), ns)
            out[k] = ns["surplus"]
            if G_base is None and "gross_income_pp" in ns:
                G_base = float(ns["gross_income_pp"])
        except Exception as exc:
            print(f"  [{tag}] t1={t1:+.3f} FAILED: {type(exc).__name__}: {exc}")
        else:
            print(f"  [{tag}] t1={t1:+.3f}  S={out[k]:+.4f}")
    return out, G_base


def make_2x2_robustness_figure(t1_grid, panels, out_path, suptitle=None):

    assert len(panels) == 4, "make_2x2_robustness_figure expects exactly 4 panels"
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 11.5))
    axes_flat = axes.flatten()
    for ax, panel in zip(axes_flat, panels):
        cmap       = panel["cmap"]
        cmap_range = panel.get("cmap_range", (0.0, 1.0))
        values     = panel["values"]
        results    = panel["results"]
        param_label = panel["param_label"]
        colors = [cmap(x) for x in np.linspace(cmap_range[0], cmap_range[1], len(values))]
        for v, color in zip(values, colors):
            ax.plot(t1_grid, results[v], linewidth=CURVE_LW,
                    marker="o", markersize=4,
                    color=color, label=f"{param_label} = {v:g}")
        ax.axhline(0.0, color="grey", linewidth=0.6)
        ax.axvline(0.25, color="seagreen", linestyle="--", linewidth=1.4)
        ax.set_xlabel(r"$\tau_1$")
        ax.set_ylabel(r"Efficiency Test  $\Delta R = R^1 - R^0$  (tDKK / person)")
        ax.set_title(panel["title"])
        ax.set_xlim(-0.1, 1.0)
        ax.grid(True, alpha=0.3)
        ax.set_box_aspect(1)
        ax.legend(loc="lower left", frameon=False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure to: {out_path}")


t1_grid       = np.linspace(-0.1, 1.0, 23)

alpha_grid    = [0.05, 0.10, 0.15, 0.20, 0.25]
epsilon_grid  = [0.48, 0.49, 0.5, 0.51, 0.52]
sigma_o_grid  = [0.2, 0.4, 0.6, 0.8, 1.0]
sigma_n_grid  = [1.2, 1.5, 1.8, 2.1, 2.4]


epsilon_phi_values = None

print("\n  Robustness 2×2 — α / ε / σ_o / σ_n sweeps")
print("  " + "="*60)

print("\n  α sweep")
print("  " + "-"*60)
res_alpha = {}
for a in alpha_grid:
    res_alpha[a], _ = sweep(t1_grid, {"alpha": a}, tag=f"α={a:.3f}")

print("\n  ε sweep")
print("  " + "-"*60)
if epsilon_phi_values is None:
    _eps_phis = [None] * len(epsilon_grid)
else:
    _eps_phis = list(epsilon_phi_values)
    if len(_eps_phis) != len(epsilon_grid):
        raise SystemExit(
            f"epsilon_phi_values must have same length as epsilon_grid "
            f"({len(epsilon_grid)} ε values vs {len(_eps_phis)} φ values)"
        )
res_eps = {}
for e, ph in zip(epsilon_grid, _eps_phis):
    overrides = {"epsilon": e}
    if ph is not None:
        overrides["phi"] = ph
    res_eps[e], _ = sweep(t1_grid, overrides, tag=f"ε={e:.2f}")

print("\n  σ_o sweep")
print("  " + "-"*60)
res_so = {}
for s in sigma_o_grid:
    res_so[s], _ = sweep(t1_grid, {"sigma_o": s}, tag=f"σ_o={s:.2f}")

print("\n  σ_n sweep")
print("  " + "-"*60)
res_sn = {}
for s in sigma_n_grid:
    res_sn[s], _ = sweep(t1_grid, {"sigma_n": s}, tag=f"σ_n={s:.2f}")



FONT_BASE = 13

plt.rcParams.update({
    "font.size":        FONT_BASE,
    "axes.titlesize":   FONT_BASE + 2,
    "axes.labelsize":   FONT_BASE + 1,
    "xtick.labelsize":  FONT_BASE - 2,
    "ytick.labelsize":  FONT_BASE - 2,
    "legend.fontsize":  FONT_BASE - 3,
    "axes.linewidth":   1.4,
})

panels = [
    {"title": r"$\alpha$ sweep (outer-nest weight)",
     "param_label": r"$\alpha$",
     "values": alpha_grid, "results": res_alpha,
     "cmap": BLUE_FADE, "cmap_range": (0.0, 1.0)},
    {"title": r"$\varepsilon$ sweep (Frisch elasticity)",
     "param_label": r"$\varepsilon$",
     "values": epsilon_grid, "results": res_eps,
     "cmap": PURPLE_FADE, "cmap_range": (0.0, 1.0)},
    {"title": r"$\sigma_o$ sweep (outer CES)",
     "param_label": r"$\sigma_o$",
     "values": sigma_o_grid, "results": res_so,
     "cmap": GREEN_FADE, "cmap_range": (0.0, 1.0)},
    {"title": r"$\sigma_n$ sweep (inner CES)",
     "param_label": r"$\sigma_n$",
     "values": sigma_n_grid, "results": res_sn,
     "cmap": ORANGE_FADE, "cmap_range": (0.0, 1.0)},
]

make_2x2_robustness_figure(
    t1_grid, panels,
    out_path=HERE / "robustness_baseline.png",
    suptitle=r"Robustness check — $\Delta R(\tau_1)$ across $\alpha$, $\varepsilon$, $\sigma_o$, $\sigma_n$",
)
