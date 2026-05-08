
import contextlib
import io
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE   = Path(__file__).parent
SCRIPT = HERE / "kaplow_becker_irs_3may_HPcal.py"

T_REFORM_RE = re.compile(r"^t_reform\s*=\s*np\.array\(\[[^\]]*\]\).*$", re.MULTILINE)


def sweep_slot(script_path: Path, slot: int, t_values: np.ndarray):

    assert slot in (0, 1, 2)
    src = script_path.read_text()
    out = np.full_like(t_values, np.nan, dtype=float)
    G_base = None
    for k, t in enumerate(t_values):
        rates = [0.25, 0.25, 0.25]
        rates[slot] = float(t)
        patched = T_REFORM_RE.sub(
            f"t_reform = np.array([{rates[0]:.10f}, {rates[1]:.10f}, {rates[2]:.10f}])",
            src, count=1,
        )
        ns = {"__name__": "__patched__", "__file__": str(script_path)}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(patched, str(script_path), "exec"), ns)
            out[k] = ns["surplus"]
            if G_base is None:
                G_base = float(ns["gross_income_pp"])
        except Exception as exc:
            print(f"  [τ{slot+1}={t:+.3f}] FAILED: {type(exc).__name__}: {exc}")
        else:
            print(f"  [τ{slot+1}={t:+.3f}]  S={out[k]:+.4f}")
    return out, G_base


t_grid = np.linspace(-0.1, 1.0, 23)

print("\n  Sweeping τ_1 (food / groceries)")
print("  " + "-"*60)
S1, G_base = sweep_slot(SCRIPT, 0, t_grid)

print("\n  Sweeping τ_2 (restaurants)")
print("  " + "-"*60)
S2, _      = sweep_slot(SCRIPT, 1, t_grid)

print("\n  Sweeping τ_3 (other goods)")
print("  " + "-"*60)
S3, _      = sweep_slot(SCRIPT, 2, t_grid)



FONT_BASE = 15

plt.rcParams.update({
    "font.size":        FONT_BASE,
    "axes.titlesize":   FONT_BASE + 2,
    "axes.labelsize":   FONT_BASE + 1,
    "xtick.labelsize":  FONT_BASE - 2,
    "ytick.labelsize":  FONT_BASE - 2,
    "legend.fontsize":  FONT_BASE - 3,
    "axes.linewidth":   1.4,
})

panel_specs = [
    (S1, r"$\tau_1$",          r"Food/grocery VAT rate $\tau_1$ (reform)"),
    (S2, r"$\tau_2$",          r"Restaurants VAT rate $\tau_2$ (reform)"),
    (S3, r"$\tau_3$",          r"Other-goods VAT rate $\tau_3$ (reform)"),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), sharey=True)

for ax, (S, sym, xlab) in zip(axes, panel_specs):
    ax.plot(t_grid, S, marker="o", linewidth=2.4, color="#dc2626",
            label="Extended Model")
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.axvline(0.25, color="seagreen", linestyle="--", linewidth=1.8,
               label=rf"{sym} = 0.25")
    ax.set_xlabel(xlab)
    ax.set_title(rf"Efficiency test vs {sym}")
    ax.set_xlim(-0.1, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")

axes[0].set_ylabel(r"Efficiency Test  $\Delta R = R^1 - R^0$  (tDKK per person)")


if G_base is not None and G_base != 0:
    ax_pct = axes[-1].secondary_yaxis(
        "right",
        functions=(lambda s: 100.0 * s / G_base,
                   lambda p: G_base * p / 100.0),
    )
    ax_pct.set_ylabel(r"$S$ as % of baseline gross income per person")

plt.tight_layout()

out = HERE / "various_extended.png"
plt.savefig(out, dpi=150)
print(f"\n  Saved figure to: {out}")
