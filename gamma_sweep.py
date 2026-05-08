import argparse
import contextlib
import io
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_GAMMA_VALUES = [0, 2, 3, 4, 5]

DEFAULT_PHI_VALUES = [1.90e+07, 1.93e+07, 1.94e+07, 1.96e+07, 1.98e+07]



BASELINE_NAVY = "#1f4e79"

GAMMA_RED      = "#dc2f26"
GAMMA_ALPHA_LO = 0.3
GAMMA_ALPHA_HI = 1.0


HERE   = Path(__file__).parent
SCRIPT = HERE / "kaplow_becker_irs_3may_HPcal.py"

T_REFORM_RE = re.compile(r"^t_reform\s*=\s*np\.array\(\[[^\]]*\]\).*$", re.MULTILINE)
SANITY_RE   = re.compile(r"^S_g0\s*=\s*_compute_S_at_gamma\b.*$",       re.MULTILINE)


def _scalar_re(name: str) -> re.Pattern:
    """Match `name = ...` assignment at the start of a line."""
    return re.compile(rf"^{re.escape(name)}\s*=.*$", re.MULTILINE)


def sweep(t1_values: np.ndarray, overrides: dict[str, float], tag: str):
    """Run the homeproduction model script once per t1 value, with module-level scalars
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


def _add_pct_axis(ax, G_base):
    """Attach a secondary y-axis showing S as % of baseline gross y/person."""
    if G_base is None or G_base == 0:
        return None
    ax_pct = ax.secondary_yaxis(
        "right",
        functions=(lambda s: 100.0 * s / G_base,
                   lambda p: G_base * p / 100.0),
    )
    ax_pct.set_ylabel(
        rf"$\Delta R$ as % of baseline gross income per person  "
        rf"($G_{{\rm base}} = {G_base:,.2f}$ tDKK)"
    )
    return ax_pct



_theta_match = re.search(r"^theta\s*=\s*([0-9.eE+-]+)",
                         SCRIPT.read_text(), re.MULTILINE)
THETA_MAIN = float(_theta_match.group(1)) if _theta_match else float("nan")

_parser = argparse.ArgumentParser(description="γ sweep at the main file's θ")
_parser.add_argument(
    "--gamma", type=float, nargs="+", default=DEFAULT_GAMMA_VALUES,
    help=f"γ values to sweep (default: {' '.join(str(g) for g in DEFAULT_GAMMA_VALUES)})",
)
_parser.add_argument(
    "--phi", type=float, nargs="+", default=DEFAULT_PHI_VALUES,
    help=("φ values, one per γ (same length as --gamma).  If omitted, "
          "the main file's φ is used for every run."),
)
_args = _parser.parse_args()

t1_grid      = np.linspace(-0.1, 1.0, 23)
gamma_values = list(_args.gamma)
if _args.phi is None:
    phi_values = [None] * len(gamma_values)
else:
    phi_values = list(_args.phi)
    if len(phi_values) != len(gamma_values):
        raise SystemExit(
            f"--phi must have same length as --gamma "
            f"({len(gamma_values)} γ values vs {len(phi_values)} φ values)"
        )

print(f"\n  γ sweep at θ = {THETA_MAIN:g} (read from {SCRIPT.name})")
res_g = {}
G_base_seen = None
for g, ph in zip(gamma_values, phi_values):
    overrides = {"gamma": g}
    if ph is not None:
        overrides["phi"] = ph
    phi_tag = f", φ = {ph:.4e}" if ph is not None else ""
    print(f"\n  Sweeping γ = {g}{phi_tag}  (θ = {THETA_MAIN:g})")
    print("  " + "-"*60)
    res_g[g], Gb = sweep(t1_grid, overrides, tag=f"γ={g:.2f}")
    if G_base_seen is None:
        G_base_seen = Gb



_pos_gamma = [g for g in gamma_values if g != 0]
_n_pos     = len(_pos_gamma)
if _n_pos > 1:
    _alpha_by_g = {
        g: GAMMA_ALPHA_LO + (GAMMA_ALPHA_HI - GAMMA_ALPHA_LO) * r / (_n_pos - 1)
        for r, g in enumerate(_pos_gamma)
    }
else:
    _alpha_by_g = {g: GAMMA_ALPHA_HI for g in _pos_gamma}

gamma_colors = [BASELINE_NAVY if g == 0 else GAMMA_RED for g in gamma_values]
gamma_alphas = [1.0 if g == 0 else _alpha_by_g[g] for g in gamma_values]


gamma_labels = [
    rf"$\gamma = {g:g}$ (Baseline / Weak-seperability)" if g == 0
    else rf"$\gamma = {g:g}$"
    for g in gamma_values
]

_n = len(gamma_values)
gamma_markers = ["o" if i == 0 or i == _n - 1 else None for i in range(_n)]


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

fig, ax = plt.subplots(figsize=(8.5, 6.0))
for v, color, alpha, lab, mk in zip(gamma_values, gamma_colors, gamma_alphas,
                                    gamma_labels, gamma_markers):
    ax.plot(t1_grid, res_g[v], linewidth=2.4,
            linestyle="-", marker=mk, markersize=5,
            color=color, alpha=alpha, label=lab)
ax.axhline(0.0, color="grey", linewidth=0.8)
ax.axvline(0.25, color="seagreen", linestyle="--", linewidth=1.8,
           label=r"$\tau_1 = 0.25$")
ax.set_xlabel(r"Food/grocery VAT rate $\tau_1$ (reform)")
ax.set_ylabel(r"Efficiency Test  $\Delta R = R^1 - R^0$  (tDKK per person)")
ax.set_title(rf"Extended model: $\Delta R$ vs $\tau_1$ across $\gamma$  ($\theta = {THETA_MAIN:g}$)")
ax.set_xlim(-0.1, 1.0)
ax.grid(True, alpha=0.3)
_add_pct_axis(ax, G_base_seen)

handles, leg_labels = ax.get_legend_handles_labels()
fig.legend(handles, leg_labels,
           loc="lower center", bbox_to_anchor=(0.5, -0.01),
           ncol=min(len(leg_labels), 3),
           frameon=False)

plt.tight_layout(rect=(0, 0.13, 1, 1))
out_path = HERE / "gamma_sweep.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\n  Saved figure to: {out_path}")
