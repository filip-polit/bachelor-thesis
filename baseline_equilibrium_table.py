import contextlib
import io
from pathlib import Path


def _import_silently(module_name):
    with contextlib.redirect_stdout(io.StringIO()):
        return __import__(module_name)


m = _import_silently("kaplow_becker_irs_3may_HPcal")

wages          = m.wages
MTR_LOW        = m.MTR_LOW
MTR_HIGH       = m.MTR_HIGH
Y_THRESHOLD    = m.Y_THRESHOLD
t_base         = m.t_base
res_base       = m.res_base
rev_base       = m.rev_base
pop            = m.pop
N              = m.N
_annual_income = m._annual_income


def fmt_num(x, decimals=1):
    s = f"{x:,.{decimals}f}"
    return s.replace(",", "{,}")


def fmt_pct(x):
    return f"{x*100:.2f}\\%"


def signed_num(x, decimals=1):
    sign = "-" if x < 0 else ""
    s = f"{abs(x):,.{decimals}f}"
    return f"{sign}{s.replace(',', '{,}')}"


rows = []
agg = dict(w=0.0, L=0.0, y=0.0, T=0.0, c=0.0,
           s1=0.0, s2=0.0, s3=0.0, U=0.0)
for i in range(N):
    r   = res_base[i]
    w   = wages[i]
    L   = r['L']
    y   = r['y']
    T   = r['T']
    c   = y - T
    U   = r['U']
    p   = [1 + t_base[j] for j in range(3)]
    M   = p[0]*r['x1'] + p[1]*r['x2'] + p[2]*r['x3']
    s1  = p[0]*r['x1'] / M
    s2  = p[1]*r['x2'] / M
    s3  = p[2]*r['x3'] / M
    mtr = MTR_LOW if y <= Y_THRESHOLD else MTR_HIGH

    rows.append((i+1, w, mtr, L, y, T, c, s1, s2, s3, U))

    agg['w']  += pop[i] * w
    agg['L']  += pop[i] * L
    agg['y']  += pop[i] * y
    agg['T']  += pop[i] * T
    agg['c']  += pop[i] * c
    agg['s1'] += pop[i] * s1
    agg['s2'] += pop[i] * s2
    agg['s3'] += pop[i] * s3
    agg['U']  += pop[i] * U


_obs_data = [
    ("Average household",           0.087087698,   0.046208038,   0.866704264),
    ("Under 250{,}000 kr.",         0.100735549,   0.047234751,   0.852029700),
    ("250{,}000--449{,}999 kr.",    0.095263750,   0.034284598,   0.870451652),
    ("450{,}000--699{,}999 kr.",    0.085543708,   0.044876892,   0.869579400),
    ("700{,}000--999{,}999 kr.",    0.089954443,   0.046207210,   0.863838348),
    ("1{,}000{,}000 kr. and above", 0.082512122,   0.050654226,   0.866833652),
]


L = []
L.append(r"\begin{table}[htbp]")
L.append(r"  \centering")
L.append(r"  \caption{Baseline (State 0) equilibrium}")
L.append(r"  \label{tab:baseline_equilibrium}")
L.append(r"  \begin{threeparttable}")
L.append(r"    \small")
L.append(r"    \setlength{\tabcolsep}{4pt}")


L.append(r"    \begin{tabular}{r r r r r r r r r}")
L.append(r"      \toprule")
L.append(r"      Decile & $L_i$ & $y_i$ & $T(y_i)$ & $c_i$ "
         r"& $s_1$ & $s_2$ & $s_3$ & $U_i$ \\")
L.append(r"             & (hrs/yr) & (tDKK) & (tDKK) & (tDKK) "
         r"& (\%) & (\%) & (\%) & \\")
L.append(r"      \midrule")
for (idx, w, mtr, Lh, y, T, c, s1, s2, s3, U) in rows:
    L.append(
        f"      {idx} & {fmt_num(Lh, 1)} & {fmt_num(y, 1)} & {signed_num(T, 1)} & "
        f"{fmt_num(c, 1)} & {fmt_pct(s1)} & {fmt_pct(s2)} & {fmt_pct(s3)} & "
        f"{fmt_num(U, 2)} \\\\"
    )
L.append(r"      \midrule")
L.append(
    f"      Pop. avg & {fmt_num(agg['L'], 1)} & {fmt_num(agg['y'], 1)} & "
    f"{signed_num(agg['T'], 1)} & {fmt_num(agg['c'], 1)} & "
    f"{fmt_pct(agg['s1'])} & {fmt_pct(agg['s2'])} & {fmt_pct(agg['s3'])} & "
    f"{fmt_num(agg['U'], 2)} \\\\"
)
L.append(r"      \midrule")
L.append(
    r"      \multicolumn{9}{l}{\textit{Total government revenue} $R^0 = "
    + fmt_num(rev_base, 2) + r"$ tDKK per person} \\"
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
out_path = OUT_DIR / "baseline_equilibrium.tex"
out_path.write_text(tex)


LB = []
LB.append(r"\begin{table}[htbp]")
LB.append(r"  \centering")
LB.append(r"  \caption{Observed gross income by decile}")
LB.append(r"  \label{tab:observed_income}")
LB.append(r"  \begin{threeparttable}")
LB.append(r"    \small")
LB.append(r"    \begin{tabular}{r r}")
LB.append(r"      \toprule")
LB.append(r"      Decile & $y$ observed (tDKK) \\")
LB.append(r"      \midrule")
for i in range(N):
    LB.append(f"      {i+1} & {fmt_num(_annual_income[i], 1)} \\\\")
LB.append(r"      \bottomrule")
LB.append(r"    \end{tabular}")
LB.append(r"    \begin{tablenotes}")
LB.append(r"      \small")
LB.append(r"      \item \textit{Notes:} % FILL IN MANUALLY")
LB.append(r"    \end{tablenotes}")
LB.append(r"  \end{threeparttable}")
LB.append(r"\end{table}")

out_path_b = OUT_DIR / "observed_income_by_decile.tex"
out_path_b.write_text("\n".join(LB) + "\n")


LC = []
LC.append(r"\begin{table}[htbp]")
LC.append(r"  \centering")
LC.append(r"  \caption{Observed expenditure shares by household income bracket}")
LC.append(r"  \label{tab:observed_expenditure_shares}")
LC.append(r"  \begin{threeparttable}")
LC.append(r"    \small")
LC.append(r"    \begin{tabular}{l r r r}")
LC.append(r"      \toprule")
LC.append(r"      Income bracket & $s_1$ (food, \%) & $s_2$ (restaurants, \%) & $s_3$ (other, \%) \\")
LC.append(r"      \midrule")
for (label, s1, s2, s3) in _obs_data:
    LC.append(f"      {label} & {fmt_pct(s1)} & {fmt_pct(s2)} & {fmt_pct(s3)} \\\\")
LC.append(r"      \bottomrule")
LC.append(r"    \end{tabular}")
LC.append(r"    \begin{tablenotes}")
LC.append(r"      \small")
LC.append(r"      \item \textit{Notes:} % FILL IN MANUALLY")
LC.append(r"    \end{tablenotes}")
LC.append(r"  \end{threeparttable}")
LC.append(r"\end{table}")

out_path_c = OUT_DIR / "observed_expenditure_shares_appendix.tex"
out_path_c.write_text("\n".join(LC) + "\n")

print(f"Baseline equilibrium table written to:  latex_output/baseline_equilibrium.tex")
print(f"Observed-income table written to:        latex_output/observed_income_by_decile.tex")
print(f"Observed-shares appendix written to:     latex_output/observed_expenditure_shares_appendix.tex")
