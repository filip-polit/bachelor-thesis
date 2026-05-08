
import contextlib
import io
from pathlib import Path


def _import_silently(module_name):
    with contextlib.redirect_stdout(io.StringIO()):
        return __import__(module_name)


m = _import_silently("kaplow_becker_irs_3may_HPcal")

wages      = m.wages
t_base     = m.t_base
t_reform   = m.t_reform
T_ref      = m.T_ref
T_tilde    = m.T_tilde
res_base   = m.res_base
res_uncomp = m.res_uncomp
res_state2 = m.res_state2
rev_base   = m.rev_base
rev_uncomp = m.rev_uncomp
rev_comp   = m.rev_comp
surplus    = m.surplus
pop        = m.pop
N          = m.N


def fmt_num(x, decimals=1):
    s = f"{x:,.{decimals}f}"
    return s.replace(",", "{,}")


def fmt_signed(x, decimals=2):
    sign = "-" if x < 0 else "+"
    s = f"{abs(x):,.{decimals}f}"
    return f"{sign}{s.replace(',', '{,}')}"


def fmt_pct(x, decimals=2):
    return f"{x*100:.{decimals}f}\\%"


def fmt_pp(x, decimals=2):
    sign = "-" if x < 0 else "+"
    return f"{sign}{abs(x)*100:.{decimals}f}"


rows1 = []
agg1 = dict(y=0., dT=0., L=0., dL=0., v=0., dv=0., U=0., dU=0.)
for i in range(N):
    r0 = res_base[i]
    r2 = res_state2[i]
    y    = r2['y']
    dT   = T_tilde(y) - T_ref(y)
    L    = r2['L']
    dL   = r2['L'] - r0['L']
    v    = r2['usub']
    dv   = r2['usub'] - r0['usub']
    U    = r2['U']
    dU   = r2['U'] - r0['U']
    rows1.append((i+1, y, dT, L, dL, v, dv, U, dU))
    agg1['y']  += pop[i] * y
    agg1['dT'] += pop[i] * dT
    agg1['L']  += pop[i] * L
    agg1['dL'] += pop[i] * dL
    agg1['v']  += pop[i] * v
    agg1['dv'] += pop[i] * dv
    agg1['U']  += pop[i] * U
    agg1['dU'] += pop[i] * dU

rows2 = []
agg2 = dict(x1=0., dx1=0., s1=0., ds1=0.)
for i in range(N):
    r0 = res_base[i]
    r2 = res_state2[i]
    x1   = r2['x1']
    dx1  = r2['x1'] - r0['x1']
    p_r  = [1 + t_reform[j] for j in range(3)]
    p_b  = [1 + t_base[j]   for j in range(3)]
    M2   = p_r[0]*r2['x1'] + p_r[1]*r2['x2'] + p_r[2]*r2['x3']
    Mb   = p_b[0]*r0['x1'] + p_b[1]*r0['x2'] + p_b[2]*r0['x3']
    s1_2 = p_r[0]*r2['x1'] / M2
    s1_b = p_b[0]*r0['x1'] / Mb
    ds1  = s1_2 - s1_b
    rows2.append((i+1, x1, dx1, s1_2, ds1))
    agg2['x1']  += pop[i] * x1
    agg2['dx1'] += pop[i] * dx1
    agg2['s1']  += pop[i] * s1_2
    agg2['ds1'] += pop[i] * ds1


L = []


L.append(r"\begin{table}[htbp]")
L.append(r"  \centering")
L.append(r"  \caption{Main results: welfare, labor, and tax adjustment under the distribution-neutral package (separable model).}")
L.append(r"  \label{tab:main_separable_welfare}")
L.append(r"  \begin{threeparttable}")
L.append(r"    \small")
L.append(r"    \setlength{\tabcolsep}{4pt}")
L.append(r"    \begin{tabular}{r r r r r r r r r}")
L.append(r"      \toprule")
L.append(r"      Decile & $y_i$ & $\Delta\tilde T$ & $L_i^{\text{S2}}$ & $\Delta L_i$ "
         r"& $v_i^{\text{S2}}$ & $\Delta v_i$ & $U_i^{\text{S2}}$ & $\Delta U_i$ \\")
L.append(r"             & (tDKK) & (tDKK) & (hrs/yr) & (hrs/yr) "
         r"&        &        &        &        \\")
L.append(r"      \midrule")
for (idx, y, dT, Lh, dL, v, dv, U, dU) in rows1:
    L.append(
        f"      {idx} & {fmt_num(y, 1)} & {fmt_signed(dT, 2)} & {fmt_num(Lh, 1)} & "
        f"{fmt_signed(dL, 2)} & {fmt_num(v, 2)} & {fmt_signed(dv, 2)} & "
        f"{fmt_num(U, 2)} & {fmt_signed(dU, 2)} \\\\"
    )
L.append(r"      \midrule")
L.append(
    f"      Pop. avg & {fmt_num(agg1['y'], 1)} & {fmt_signed(agg1['dT'], 2)} & "
    f"{fmt_num(agg1['L'], 1)} & {fmt_signed(agg1['dL'], 2)} & "
    f"{fmt_num(agg1['v'], 2)} & {fmt_signed(agg1['dv'], 2)} & "
    f"{fmt_num(agg1['U'], 2)} & {fmt_signed(agg1['dU'], 2)} \\\\"
)
L.append(r"      \midrule")
L.append(
    r"      \multicolumn{9}{l}{$R^0 = " + fmt_num(rev_base, 2)
    + r"$, $R^1 = " + fmt_num(rev_comp, 2)
    + r"$, $S = R^1 - R^0 = " + fmt_signed(surplus, 2)
    + r"$ (tDKK per person)} \\"
)
L.append(r"      \bottomrule")
L.append(r"    \end{tabular}")
L.append(r"    \begin{tablenotes}")
L.append(r"      \small")
L.append(r"      \item \textit{Notes:} % FILL IN MANUALLY")
L.append(r"    \end{tablenotes}")
L.append(r"  \end{threeparttable}")
L.append(r"\end{table}")

L.append(r"")


L.append(r"\begin{table}[htbp]")
L.append(r"  \centering")
L.append(r"  \caption{Food consumption response under the distribution-neutral package (separable model).}")
L.append(r"  \label{tab:main_separable_food}")
L.append(r"  \begin{threeparttable}")
L.append(r"    \small")
L.append(r"    \setlength{\tabcolsep}{4pt}")
L.append(r"    \begin{tabular}{r r r r r}")
L.append(r"      \toprule")
L.append(r"      Decile & $x_1^{\text{S2}}$ & $\Delta x_1$ & $s_1^{\text{S2}}$ & $\Delta s_1$ \\")
L.append(r"             & (tDKK)            & (tDKK)       & (\%)              & (pp)         \\")
L.append(r"      \midrule")
for (idx, x1, dx1, s1, ds1) in rows2:
    L.append(
        f"      {idx} & {fmt_num(x1, 2)} & {fmt_signed(dx1, 2)} & "
        f"{fmt_pct(s1, 2)} & {fmt_pp(ds1, 2)} \\\\"
    )
L.append(r"      \midrule")
L.append(
    f"      Pop. avg & {fmt_num(agg2['x1'], 2)} & {fmt_signed(agg2['dx1'], 2)} & "
    f"{fmt_pct(agg2['s1'], 2)} & {fmt_pp(agg2['ds1'], 2)} \\\\"
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
out_path = OUT_DIR / "main_results_separable.tex"
out_path.write_text(tex)

print(f"Main results tables (separable) written to: latex_output/main_results_separable.tex")
