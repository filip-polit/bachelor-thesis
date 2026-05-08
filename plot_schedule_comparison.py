import contextlib
import io
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _import_silently(module_name):
    with contextlib.redirect_stdout(io.StringIO()):
        return __import__(module_name)


m = _import_silently("kaplow_becker_irs_3may_HPcal copy 2")

y_grid       = m.y_grid
T_tilde_vals = m.T_tilde_vals
T_ref        = m.T_ref
Y_THRESHOLD  = m.Y_THRESHOLD
res_base     = m.res_base
N            = m.N


HERE    = Path(__file__).parent
FIG_DIR = HERE / "latex_output"
FIG_DIR.mkdir(exist_ok=True)

_y_plot_max = 1400
_mask       = y_grid <= _y_plot_max
_y_plot     = y_grid[_mask]
_T_tilde_p  = T_tilde_vals[_mask]
_T_ref_p    = np.array([T_ref(y) for y in _y_plot])

_decile_y = np.array([r['y'] for r in res_base])

fig, axL = plt.subplots(figsize=(6.5, 4.5))

axL.axhline(0.0, color="0.4", linewidth=0.6, zorder=1)
axL.axvline(Y_THRESHOLD, color="0.5", linestyle="--", linewidth=0.7,
            alpha=0.8, zorder=1)
axL.plot(_y_plot, _T_ref_p, color="black", linewidth=1.1,
         label=r"$\bar T(y)$ reference DK income tax schedule")
axL.plot(_y_plot, _T_tilde_p, color="#3145ca", linewidth=1.1,
         linestyle="-", zorder=3,
         label=r"$ T^\circ(y)$ DN income tax schedule")
_T_ref_at_decile = np.array([T_ref(y) for y in _decile_y])
axL.scatter(_decile_y, _T_ref_at_decile,
            s=8, color="black", zorder=4,
            label="decile state 0 positions")
axL.set_xlim(0, _y_plot_max)
axL.set_xlabel("Gross income $y$ (tDKK/year)")
axL.set_ylabel("Net tax (tDKK/year)")
axL.set_title("Income-tax schedule: $\\bar  T(y)$ vs $ T^\circ(y)$")
axL.legend(loc="upper left", frameon=False, fontsize=9)

plt.tight_layout()
_pdf_path = FIG_DIR / "schedule_comparison.pdf"
_png_path = FIG_DIR / "schedule_comparison.png"
plt.savefig(_pdf_path)
plt.savefig(_png_path, dpi=150)
plt.close(fig)


_diff = _T_tilde_p - _T_ref_p

fig_d, axD = plt.subplots(figsize=(6.5, 4.5))

axD.axhline(0.0, color="0.4", linewidth=0.6, zorder=1)
axD.axvline(Y_THRESHOLD, color="0.5", linestyle="--", linewidth=0.7,
            alpha=0.8, zorder=1)
axD.plot(_y_plot, _diff, color="#1f4e79", linewidth=2.4, zorder=3,
         label=r"$T^\circ(y) - \bar T(y)$")

_diff_at_decile = np.array([
    float(np.interp(y, _y_plot, _diff)) for y in _decile_y
])
axD.scatter(_decile_y, _diff_at_decile,
            s=15, color="#14324e", zorder=4,
            label="decile state 0 positions")

axD.set_xlim(0, _y_plot_max)
axD.set_xlabel("Gross income $y$ (tDKK/year)")
axD.set_ylabel(r"$T^\circ(y) - \bar T(y)$ (tDKK/year)")
axD.set_title(r"Schedule difference: $T^\circ(y) - \bar T(y)$")
axD.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
           ncol=2, frameon=False, fontsize=9)

plt.tight_layout()
_pdf_diff = FIG_DIR / "schedule_difference.pdf"
_png_diff = FIG_DIR / "schedule_difference.png"
plt.savefig(_pdf_diff, bbox_inches="tight")
plt.savefig(_png_diff, dpi=150, bbox_inches="tight")
plt.close(fig_d)


print(f"Schedule-comparison figure written to:")
print(f"  {_pdf_path}")
print(f"  {_png_path}")
print(f"Schedule-difference figure written to:")
print(f"  {_pdf_diff}")
print(f"  {_png_diff}")


fig_c, (axCL, axCR) = plt.subplots(1, 2, figsize=(11.5, 5.5))

axCL.axhline(0.0, color="0.4", linewidth=0.6, zorder=1)
axCL.axvline(Y_THRESHOLD, color="0.5", linestyle="--", linewidth=0.7,
             alpha=0.8, zorder=1)
axCL.plot(_y_plot, _T_ref_p, color="black", linewidth=1.1,
          label=r"$\bar T(y)$ reference DK income tax schedule")
axCL.plot(_y_plot, _T_tilde_p, color="#dc2626", linewidth=1.1,
          linestyle="-", zorder=3,
          label=r"$T^\circ(y)$ DN income tax schedule")
axCL.scatter(_decile_y, _T_ref_at_decile,
             s=8, color="black", zorder=4,
             label="decile state 0 positions")
axCL.set_xlim(0, _y_plot_max)
axCL.set_xlabel("Gross income $y$ (tDKK/year)")
axCL.set_ylabel("Net tax (tDKK/year)")
axCL.set_title(r"Income-tax schedule: $\bar T(y)$ vs $T^\circ(y)$")
axCL.legend(loc="upper left", frameon=False, fontsize=9)
axCL.set_box_aspect(1)

axCR.axhline(0.0, color="0.4", linewidth=0.6, zorder=1)
axCR.axvline(Y_THRESHOLD, color="0.5", linestyle="--", linewidth=0.7,
             alpha=0.8, zorder=1)
axCR.plot(_y_plot, _diff, color="#dc2626", linewidth=1.1, zorder=3,
          label=r"$T^\circ(y) - \bar T(y)$")
axCR.scatter(_decile_y, _diff_at_decile,
             s=8, color="black", zorder=4,
             label="decile state 0 positions")
axCR.set_xlim(0, _y_plot_max)
axCR.set_xlabel("Gross income $y$ (tDKK/year)")
axCR.set_ylabel(r"$T^\circ(y) - \bar T(y)$ (tDKK/year)")
axCR.set_title(r"Schedule difference: $T^\circ(y) - \bar T(y)$")
axCR.legend(loc="lower left", frameon=False, fontsize=9)
axCR.set_box_aspect(1)

plt.tight_layout()
_pdf_combined = FIG_DIR / "schedule_comparison_combined.pdf"
_png_combined = FIG_DIR / "schedule_comparison_combined.png"
plt.savefig(_pdf_combined)
plt.savefig(_png_combined, dpi=150)
plt.close(fig_c)

print(f"Schedule-comparison combined figure written to:")
print(f"  {_pdf_combined}")
print(f"  {_png_combined}")
