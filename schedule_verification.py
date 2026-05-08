
import contextlib
import io
from pathlib import Path


def _import_silently(module_name):
    with contextlib.redirect_stdout(io.StringIO()):
        return __import__(module_name)


m = _import_silently("kaplow_becker_irs_3may_HPcal")

t_base        = m.t_base
t_reform      = m.t_reform
T_ref         = m.T_ref
T_tilde       = m.T_tilde
w_of_y        = m.w_of_y
_U_at_fixed_L = m._U_at_fixed_L


def fmt_num(x, decimals=1):
    s = f"{x:,.{decimals}f}"
    return s.replace(",", "{,}")


def fmt_signed(x, decimals=2):
    sign = "-" if x < 0 else "+"
    s = f"{abs(x):,.{decimals}f}"
    return f"{sign}{s.replace(',', '{,}')}"



y_grid = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000]

rows = []
max_abs_dU = 0.0
for y in y_grid:
    w   = w_of_y(y)
    L_M = y / w if w > 0 else 0.0
    c0  = y - T_ref(y)
    c2  = y - T_tilde(y)
    U0 = _U_at_fixed_L(tuple(t_base),   c0, L_M)
    U1 = _U_at_fixed_L(tuple(t_reform), c0, L_M)
    U2 = _U_at_fixed_L(tuple(t_reform), c2, L_M)
    if U0 is None or U2 is None:
        dU = float("nan")
    else:
        dU = U2 - U0
        if abs(dU) > max_abs_dU:
            max_abs_dU = abs(dU)
    rows.append((y, U0, U1, U2, dU))



L = []
L.append(r"\begin{table}[htbp]")
L.append(r"  \centering")
L.append(r"  \caption{Verification: utility under each regime across the income grid (separable model).}")
L.append(r"  \label{tab:schedule_verification}")
L.append(r"  \begin{threeparttable}")
L.append(r"    \small")
L.append(r"    \setlength{\tabcolsep}{4pt}")
L.append(r"    \begin{tabular}{r r r r r}")
L.append(r"      \toprule")
L.append(r"      $y$ & $U^{(0)}$ & $U^{(1)}$ & $U^{(2)}$ & $\Delta U$ \\")
L.append(r"      (tDKK) &           &           &           &            \\")
L.append(r"      \midrule")
for (y, U0, U1, U2, dU) in rows:
    U0_s = fmt_num(U0, 6) if U0 is not None else "--"
    U1_s = fmt_num(U1, 6) if U1 is not None else "--"
    U2_s = fmt_num(U2, 6) if U2 is not None else "--"
    dU_s = fmt_signed(dU, 6) if dU == dU else "--"  # NaN check
    L.append(
        f"      {fmt_num(y, 0)} & {U0_s} & {U1_s} & {U2_s} & {dU_s} \\\\"
    )
L.append(r"      \midrule")
L.append(
    r"      \multicolumn{5}{l}{Max $|\Delta U|$ across grid = "
    + fmt_num(max_abs_dU, 6)
    + r", computed as $\max_y |U^{(2)}(y) - U^{(0)}(y)|$.} \\"
)
L.append(r"      \bottomrule")
L.append(r"    \end{tabular}")
L.append(r"    \begin{tablenotes}")
L.append(r"      \small")
L.append(r"      \item \textit{Notes:} % FILL IN MANUALLY")
L.append(r"    \end{tablenotes}")
L.append(r"  \end{threeparttable}")
L.append(r"\end{table}")

tex = "\n".join(L) + "\n"

HERE    = Path(__file__).parent
OUT_DIR = HERE / "latex_output"
OUT_DIR.mkdir(exist_ok=True)
out_path = OUT_DIR / "schedule_verification.tex"
out_path.write_text(tex)

print(f"Schedule-verification table written to: latex_output/schedule_verification.tex")
