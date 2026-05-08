
import contextlib
import io
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

HERE   = Path(__file__).parent
SCRIPT = HERE / "kaplow_becker_irs_3may_HPcal.py"

T_REFORM_RE = re.compile(r"^t_reform\s*=\s*np\.array\(\[[^\]]*\]\).*$", re.MULTILINE)


def sweep(script_path: Path, t1_values: np.ndarray):

    src = script_path.read_text()
    out = np.full_like(t1_values, np.nan, dtype=float)
    G_base = None
    for k, t1 in enumerate(t1_values):
        patched = T_REFORM_RE.sub(
            f"t_reform = np.array([{t1:.10f}, 0.25, 0.25])",
            src,
            count=1,
        )
        ns = {"__name__": "__patched__", "__file__": str(script_path)}
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compile(patched, str(script_path), "exec"), ns)
            out[k] = ns["surplus"]
            if G_base is None:
                G_base = float(ns["gross_income_pp"])
        except Exception as exc:
            print(f"  [{script_path.name}] t1={t1:+.3f} FAILED: {type(exc).__name__}: {exc}")
        else:
            print(f"  [{script_path.name}] t1={t1:+.3f}  S={out[k]:+.4f}")
    return out, G_base


t1_grid = np.linspace(-0.1, 1.0, 23)

print("\n  Sweeping kaplow_schedule_construction_freewage.py (baseline / separable)")
print("  " + "-"*60)
S, G_base = sweep(SCRIPT, t1_grid)


# Plot --------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))

ax.plot(t1_grid, S, marker="o", linewidth=2.4, color="#1f4e79",
        label="baseline model (weak-seperability)")
ax.axhline(0.0, color="grey", linewidth=0.8)
ax.axvline(0.25, color="seagreen", linestyle="--", linewidth=1.8,
           label="$t_1 = 0.25$")
ax.set_xlabel("Food/grocery VAT rate $\\tau_1$ (reform)")
ax.set_ylabel("Efficiency Test  $\\Delta R = R_2 - R_0$  (tDKK per person)")
ax.set_title("Efficiency test vs reform $\\tau_1$:  $\\tau_1 \\in [-0.1, 1.0]$")
ax.set_xlim(-0.1, 1.0)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower left")

if G_base is not None and G_base != 0:
    ax_pct = ax.secondary_yaxis(
        "right",
        functions=(lambda s: 100.0 * s / G_base,
                   lambda p: G_base * p / 100.0),
    )
    ax_pct.set_ylabel(
        rf"$S$ as % of baseline gross income per person  "
    )

plt.tight_layout()

out = HERE / "baseline_various.png"
plt.savefig(out, dpi=150)
print(f"\n  Saved figure to: {out}")
plt.show()
