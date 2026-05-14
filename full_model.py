# BACHELOR-THESIS
# SIGURD DYBBRO RING & FILIP BUDNY (2026)
# DIFFERENTIATED VAT IN DENMARK: AN EFFICIENCY ANALYSIS

import numpy as np
from scipy.optimize import brentq, minimize_scalar
import warnings
warnings.filterwarnings("ignore")


HOURS_YEAR = 1_380.0 #average hours worked per year (calibration)
N   = 10 # number of wage types
pop = np.full(N, 0.10) # pop shares

# observed annual average incomes, only relevant for calibration
_annual_income = np.array([
      65.300,  165.400,
     216.100,  261.300,
     314.900,  385.700,
     458.700,  539.000,
     657.900, 1311.600,
])
# wages of each type (calibrated to match obs incomes)
wages = np.array([
    0.083, 0.154, 0.184, 0.208, 0.236,
    0.270, 0.303, 0.337, 0.384, 0.675,
])


t_base   = np.array([0.25, 0.25, 0.25]) # status quo vat policy
t_reform = np.array([0.0,  0.25, 0.25]) # reform vat policy


#outer nest parameters
sigma_n = 1.5 #inner-nest elasticity (food vs restaurants)
sigma_o = 0.6 #outer-nest elasticity (nutrition vs other)
alpha   = 0.15 #outer-nest weight on nutrition composite (vs other goods)
epsilon = 0.5 #frisch elasticity

_obs_shares = np.array([
    [0.101, 0.0470, 0.852],
    [0.101, 0.0470, 0.852],
    [0.101, 0.0470, 0.852],
    [0.095, 0.0340, 0.870],
    [0.095, 0.0340, 0.870],
    [0.095, 0.0340, 0.870],
    [0.086, 0.0450, 0.870],
    [0.086, 0.0450, 0.870],
    [0.086, 0.0450, 0.870],
    [0.083, 0.0510, 0.867],
])
_obs_shares = _obs_shares / _obs_shares.sum(axis=1, keepdims=True)

p_calib = 1.0 + t_base

#inner nest parameters
beta = np.array([1.5, 1.1]) #inner-nest weights
k    = np.array([2.8,  0,  0. ]) #subsistence levels
phi  = 1.98e+07 #labor scaler

gamma = 5 # home production labor and strain per unit of real food
theta = 0.5 # returns of scale in home production

print(f"\n  Manual calibration parameters:")
print(f"    beta    = [{beta[0]:.6f}, {beta[1]:.6f}]  (inner-nest weights, food/rest.)")
print(f"    alpha   = {alpha:.4f}  (outer weight on nutrition composite)")
print(f"    k       = [{k[0]:,.1f}, {k[1]:,.1f}, {k[2]:.1f}]  tDKK/year")
print(f"    phi     = {phi:.6e}  (manually set; check avg L_M below)")
print(f"    sigma_n = {sigma_n}  (food <-> restaurants)")
print(f"    sigma_o = {sigma_o}  (nutrition <-> other)")
print(f"    epsilon = {epsilon}")
print(f"  Home-labor technology: L_H = gamma * x1**theta")
print(f"    gamma   = {gamma} hours per unit real food (1 unit = 1 tDKK pre-VAT)")
print(f"    theta = {theta}  ({'linear (CRS)' if theta == 1.0 else 'increasing returns'})")

# subutility function
def subutility(x1, x2, x3):
    b1, b2 = beta
    k1, k2, k3 = k
    c1 = x1 - k1; c2 = x2 - k2; c3 = x3 - k3 #c is consumption over subsidence level
    if c1 <= 0 or c2 <= 0 or c3 <= 0: # ensures that  actual consumption is always bigger than subsistence levels
        return -1e10
    rho_n = (sigma_n - 1) / sigma_n # converts elasticities to CES variables
    rho_o = (sigma_o - 1) / sigma_o # converts elasticities to CES variables
    N = (b1 * c1**rho_n + b2 * c2**rho_n)**(1 / rho_n) # N is the inner-nest nutrition nest
    return (alpha * N**rho_o + (1 - alpha) * c3**rho_o)**(1 / rho_o) # returns outer-nest CES utility over (groceries/restaurants)and over goods

# utility function
def utility(x1, x2, x3, L_M):
    """U = u(x) − v(L_total),  L_total = L_M + gamma · x1^theta  (Becker + IRS)."""
    c = subutility(x1, x2, x3) 
    if c <= -1e9:
        return -1e10 # computes goods subutility, returns very low subutility if it is unfeasable.
    L_H = gamma * x1**theta if x1 > 0 else 0.0
    L_total = L_M + L_H #home labou rrequired to procue x_1 food, and total labour (market + home)
    if L_total <= 0:
        return c
    v_L = L_total**(1 + 1/epsilon) / ((1 + 1/epsilon) * phi)
    return c - v_L #isoelastic labour disutility consistent with frisch elasticity

# nested aggregates
def _nested_aggregates(p1, p2, p3):
    """Return (W_n, P_N, W_o, mu) at consumer prices (p1, p2, p3)."""
    b1, b2 = beta
    W_n = b1**sigma_n * p1**(1 - sigma_n) + b2**sigma_n * p2**(1 - sigma_n) #inner-nest price aggregator
    P_N = W_n**(1 / (1 - sigma_n)) # inner-nest price index for the nutrition nest
    W_o = alpha**sigma_o * P_N**(1 - sigma_o) + (1 - alpha)**sigma_o * p3**(1 - sigma_o) # outer-nest price aggreagotor
    mu  = W_o**(1 / (sigma_o - 1)) #marginal utility of supernumerary expenditure
    return W_n, P_N, W_o, mu

# nested demands
def _nested_demands(M_sup, p1, p2, p3, W_n, P_N, W_o):
    """Stage-2 then stage-1 split of supernumerary expenditure into x1, x2, x3."""
    b1, b2 = beta
    E_N = (alpha**sigma_o * P_N**(1 - sigma_o) / W_o) * M_sup
    E_3 = M_sup - E_N # stage-1 split: supernumerary expenditure M^sup allocated between nutrition composite E_N and other goods E_3 via CES demand
    x1 = k[0] + (b1**sigma_n * p1**(-sigma_n) / W_n) * E_N
    x2 = k[1] + (b2**sigma_n * p2**(-sigma_n) / W_n) * E_N
    x3 = k[2] + E_3 / p3
    return x1, x2, x3 # stage-2 split: nutrition composite E_N allocated between x1 and x2 via CES demand

#equillibrium solver
def solve_consumer(wage, mtr, T, t1, t2, t3):
    """Solve the household's utility-maximization problem under nested CES with
    Becker home labour and IRS in home production."""
    p1, p2, p3 = 1 + t1, 1 + t2, 1 + t3
    p   = np.array([p1, p2, p3])
    wn  = (1 - mtr) * wagen # consumer prices and after-tax wage given a linear MTR

    W_n, P_N, W_o, _ = _nested_aggregates(p1, p2, p3)
    pk = p @ k # price aggregators and cost of lowest possible bunde (subsistence bundle)


# negative utility: A function of market labor. Returns a large penalty if infeasible. Cash on hand M = after-tax labour income plus lump-sum transfer T. Then subtracts subsistence costs.
    def _neg_U(L_M):
        if L_M <= 0:
            return 1e10
        M     = wn * L_M + T
        M_sup = M - pk
        if M_sup <= 1e-6:
            return 1e10
        x1, x2, x3 = _nested_demands(M_sup, p1, p2, p3, W_n, P_N, W_o)
        U = utility(x1, x2, x3, L_M)
        if not np.isfinite(U):
            return 1e10
        return -U

    opt = minimize_scalar(_neg_U,
                          bounds=(1e-6, HOURS_YEAR * 5.0),
                          method='bounded',
                          options={'xatol': 1e-6, 'maxiter': 300})
    L_M = opt.x # bounded 1 dimensional search for optimal market labor (see section 4)
    M     = wn * L_M + T
    M_sup = M - pk
    if M_sup <= 1e-6 or L_M <= 0:
        return dict(x1=0., x2=0., x3=0., L=0., Y=0., U=-1e10)
    x1, x2, x3 = _nested_demands(M_sup, p1, p2, p3, W_n, P_N, W_o)
    U = utility(x1, x2, x3, L_M)
    return dict(x1=x1, x2=x2, x3=x3, L=L_M, Y=wage*L_M, U=U)

# equillibrium solver for fixed L_M. Same structure as above.
def solve_consumer_fixed_L(wage, mtr, T, t1, t2, t3, L_fixed):
    """Demand allocation at given prices and transfer, holding L_M = L_fixed."""
    p1, p2, p3 = 1 + t1, 1 + t2, 1 + t3
    p   = np.array([p1, p2, p3])
    wn  = (1 - mtr) * wage

    W_n, P_N, W_o, _ = _nested_aggregates(p1, p2, p3)

    M     = wn * L_fixed + T
    M_sup = M - p @ k
    if M_sup <= 1e-6:
        return dict(x1=0., x2=0., x3=0., L=L_fixed, Y=wage*L_fixed, U=-1e10)

    x1, x2, x3 = _nested_demands(M_sup, p1, p2, p3, W_n, P_N, W_o)
    U = utility(x1, x2, x3, L_fixed)
    return dict(x1=x1, x2=x2, x3=x3, L=L_fixed, Y=wage*L_fixed, U=U)

# government revenue function: per capita government revenue: population-weighted sum of net incomet ax paid plus commodity tax revenue on each good. 
def gov_revenue(results, t1, t2, t3):
    """Sum of income tax (results[i]['T']) plus commodity tax revenue.
    Requires `results` to come from optimize_household."""
    return sum(
        pop[i] * (results[i]['T']
                  + t1*results[i]['x1'] + t2*results[i]['x2'] + t3*results[i]['x3'])
        for i in range(N)
    )


G = 148.33 # sets amount of public good expenditure per person


def print_shares(results, t1, t2, t3, label=""): # simply print implied expenditure shares
    p = np.array([1+t1, 1+t2, 1+t3])
    if label:
        print(f"\n  Consumption shares — {label}")
    print(f"  {'Type':>5}  {'s1 (food)':>10}  {'s2 (rest.)':>11}  {'s3 (other)':>12}")
    for i in range(N):
        r = results[i]
        M = p[0]*r['x1'] + p[1]*r['x2'] + p[2]*r['x3']
        s = [p[j]*r[f'x{j+1}']/M for j in range(3)]
        print(f"  {i+1:>5}  {s[0]:>10.2%}  {s[1]:>11.2%}  {s[2]:>12.2%}")

# tax schedule definiiton: two-bracket piecewise-linear income tax 45 % up to the kink at 750 tDKK, 60 % above. Mimicking danish income tax system.
MTR_LOW   = 0.45 # low bracket
MTR_HIGH  = 0.60 # high bracket
Y_THRESHOLD = 750 #threshold (kink)
T_AT_THRESHOLD = MTR_LOW * Y_THRESHOLD

 
def MTR_of_y(y):  # returns marginal tax rate, at income y
    """Marginal tax rate at gross income y, derived from T_ref."""
    return MTR_LOW if y <= Y_THRESHOLD else MTR_HIGH


def _T_no_lump(y): # income tax paid before the lump-sum transfer
    """Income tax paid (gross of any lump-sum transfer)."""
    if y <= Y_THRESHOLD:
        return MTR_LOW * y
    return T_AT_THRESHOLD + MTR_HIGH * (y - Y_THRESHOLD)


def indirect_subutility(t1, t2, t3, c): #indirect subutility given total disposible income c (no labour disutility)
    """Indirect subutility from total cash c at consumer prices p_j = 1+t_j."""
    p = np.array([1+t1, 1+t2, 1+t3])
    M_sup = c - p @ k
    if M_sup <= 1e-9:
        return -1e10
    W_n, P_N, W_o, _ = _nested_aggregates(*p)
    x1, x2, x3 = _nested_demands(M_sup, *p, W_n, P_N, W_o)
    return subutility(x1, x2, x3)


def consumer_at_cash(t1, t2, t3, c): #same as above, but returns demand bundle along with the subutility (used ofr optimize_household)
    """Demand allocation and subutility given cash budget c at prices p_j = 1+t_j."""
    p = np.array([1+t1, 1+t2, 1+t3])
    M_sup = c - p @ k
    if M_sup <= 1e-9:
        return None
    W_n, P_N, W_o, _ = _nested_aggregates(*p)
    x1, x2, x3 = _nested_demands(M_sup, *p, W_n, P_N, W_o)
    return dict(x1=x1, x2=x2, x3=x3, M_sup=M_sup,
                usub=subutility(x1, x2, x3))

#optimize household: Inner objective: for L_M compute gross income y, tax T from the supplied schedule, disposable income, demands, home labor, total labor, disutility and full utility.
def optimize_household(wage, prices, schedule):
    """Maximise full utility U(L_M) for a household, given the after-tax wage,
    a 3-tuple of commodity tax rates (t1, t2, t3), and a callable income-tax
    schedule schedule(y)."""
    def _neg_U(L_val):
        if L_val <= 0:
            return 1e10
        y    = wage * L_val
        T_t  = schedule(y)
        c    = y - T_t
        info = consumer_at_cash(*prices, c)
        if info is None:
            return 1e10
        x1_v = info['x1']
        L_H  = gamma * x1_v**theta if x1_v > 0 else 0.0
        L_total = L_val + L_H
        v_L = L_total**(1 + 1/epsilon) / ((1 + 1/epsilon) * phi)
        U = info['usub'] - v_L
        return -U

# solve and recompute at optimum
    opt = minimize_scalar(_neg_U,
                          bounds=(1.0, HOURS_YEAR * 5.0),
                          method='bounded',
                          options={'xatol': 1e-6, 'maxiter': 300})
    L_M = opt.x
    y = wage * L_M
    T_t = schedule(y)
    c = y - T_t
    info = consumer_at_cash(*prices, c)
    if info is None:
        return dict(L=L_M, y=y, T=T_t, c=c, x1=0., x2=0., x3=0.,
                    usub=-1e10, U=-1e10, Y=wage*L_M)
    x1_v = info['x1']
    L_H  = gamma * x1_v**theta if x1_v > 0 else 0.0
    L_total = L_M + L_H
    v_L = L_total**(1 + 1/epsilon) / ((1 + 1/epsilon) * phi)
    U = info['usub'] - v_L
    return dict(L=L_M, y=y, T=T_t, c=c,
                x1=info['x1'], x2=info['x2'], x3=info['x3'],
                usub=info['usub'], U=U, Y=wage*L_M)

# lump sum transfer calibration:
# runs baseline with no transfer to get gross revenue, then sets T_lump so that net revenue exactly funds G per person. Rho_implied is the fraction of gross revenue retained for G.
# the rest gets returned to households
_res_gross_0 = [optimize_household(wages[i], tuple(t_base), _T_no_lump) for i in range(N)]
rev_gross_0  = gov_revenue(_res_gross_0, *t_base)

T_lump       = rev_gross_0 - G
T_fixed      = np.full(N, T_lump)
rho_implied  = G / rev_gross_0 if rev_gross_0 != 0 else float('nan')

# net tax schedule: income tax minus lump-sum transfer (reference schedule used in state 0 ands state 1)
def T_ref(y):
    """Reference net tax schedule (income tax paid minus lump-sum transfer)."""
    return _T_no_lump(y) - T_lump


print(f"\n  Budget specification:")
print(f"    G (public-goods requirement) : {G:>12,.2f} tDKK per person  [absolute]")
print(f"    rev_gross_0 (gross baseline) : {rev_gross_0:>12,.2f} tDKK per person")
print(f"    Implied retention share rho  : {rho_implied:>12.1%}  [G / rev_gross_0, for reference]")
print(f"    T_lump (per-capita transfer) : {T_lump:>12,.2f} tDKK per person  [rev_gross_0 - G]")

print(f"\n  Labour supply at baseline (with T_ref schedule applied):")
print(f"  {'Type':>5}  {'w (tDKK/hr)':>12}  {'y (tDKK/yr)':>12}  {'MTR':>6}  {'L_M (hours)':>12}")
for i in range(N):
    r = optimize_household(wages[i], tuple(t_base), T_ref)
    print(f"  {i+1:>5}  {wages[i]:>12.2f}  {r['y']:>12.2f}  "
          f"{MTR_of_y(r['y']):>6.1%}  {r['L']:>12.2f}")
_L_baseline_top = [optimize_household(wages[i], tuple(t_base), T_ref)['L'] for i in range(N)]
_L_avg_top = sum(pop[i] * _L_baseline_top[i] for i in range(N))
print(f"  Population-weighted average L_M: {_L_avg_top:.2f} hours/year  (target: {HOURS_YEAR:.0f})")


print(f"\n  KAPLOW REFORM (BECKER HOME LABOUR + IRS) — Filip Budny & Sigurd Dybbro Ring (2026)")
print(f"  sigma_n={sigma_n}  sigma_o={sigma_o}  alpha={alpha}  epsilon={epsilon}  phi={phi:.4e}")
print(f"  beta=[{beta[0]:.4f}, {beta[1]:.4f}]  (inner-nest, food/rest.)")
print(f"  k   =[{k[0]:,.1f}, {k[1]:,.1f}, {k[2]:.1f}] tDKK/year")
print(f"  gamma   = {gamma}  (hours per unit real food; 1 unit = 1 tDKK at producer prices)")
print(f"  theta = {theta}  ({'linear (CRS)' if theta == 1.0 else 'increasing returns'})")
print(f"  Baseline VAT (State 0)   : t1={t_base[0]:.0%}  t2={t_base[1]:.0%}  t3={t_base[2]:.0%}  [uniform]")
print(f"  Reform   VAT (States 1-2): t1={t_reform[0]:.0%}  t2={t_reform[1]:.0%}  t3={t_reform[2]:.0%}  [food zero-rated]")

hdr = (f"  {'Type':>5}  {'w':>6}  {'L_M':>8}  {'y':>8}  {'T(y)':>9}  "
       f"{'x1':>9}  {'x2':>9}  {'x3':>10}  {'usub':>10}  {'U':>10}")

print(f"\n  Running with γ = {gamma}, θ = {theta}.  "
      f"(γ = 0 collapses to the separable model regardless of θ.)")

print(f"\n  STATE 0 — Baseline (uniform 25% VAT, schedule T_ref(y); γ = {gamma}, θ = {theta})")
# STATE 0 / STATUS QUO EQUILLIBRIUM:
# SOLVES EACH INDIVIDUAL PROBLEM AT STATUS QUO VAT PRICES AND REFERENCE SCHEDULE
# REPORTS BACK UTILITIES, HOURS SUBUTILITIES AND REVENUE R^0
res_base = [optimize_household(wages[i], tuple(t_base), T_ref) for i in range(N)]
U_base   = [r['U']    for r in res_base]
L_base   = [r['L']    for r in res_base]
usub_base= [r['usub'] for r in res_base]
rev_base = sum(pop[i] * (res_base[i]['T']
                          + t_base[0] * res_base[i]['x1']
                          + t_base[1] * res_base[i]['x2']
                          + t_base[2] * res_base[i]['x3'])
               for i in range(N))

print(f"  Government revenue: {rev_base:+,.2f} tDKK per person")
print(hdr)
for i in range(N):
    r = res_base[i]
    print(f"  {i+1:>5}  {wages[i]:>6.2f}  {r['L']:>8.2f}  {r['y']:>8.2f}  {r['T']:>+9.2f}  "
          f"{r['x1']:>9.2f}  {r['x2']:>9.2f}  {r['x3']:>10,.0f}  "
          f"{r['usub']:>10.4f}  {r['U']:>10.4f}")
print_shares(res_base, *t_base, label="State 0 — Baseline")

print(f"\n  STATE 1 — Intermediate ΔP only (reform VAT, schedule T_ref unchanged)")
# STATE 1 - ONLY VAT REFORM - 
# SOLVES EQUILLIBRIUM NOW WITH CHANGE TO PRICES
res_uncomp = [optimize_household(wages[i], tuple(t_reform), T_ref) for i in range(N)]
rev_uncomp = sum(pop[i] * (res_uncomp[i]['T']
                            + t_reform[0] * res_uncomp[i]['x1']
                            + t_reform[1] * res_uncomp[i]['x2']
                            + t_reform[2] * res_uncomp[i]['x3'])
                 for i in range(N))

print(f"  Government revenue: {rev_uncomp:+,.2f} tDKK per person")
print(f"  Revenue change vs baseline: {rev_uncomp - rev_base:+,.2f} tDKK per person")
print(hdr)
for i in range(N):
    r = res_uncomp[i]
    print(f"  {i+1:>5}  {wages[i]:>6.2f}  {r['L']:>8.2f}  {r['y']:>8.2f}  {r['T']:>+9.2f}  "
          f"{r['x1']:>9.2f}  {r['x2']:>9.2f}  {r['x3']:>10,.0f}  "
          f"{r['usub']:>10.4f}  {r['U']:>10.4f}")
print_shares(res_uncomp, *t_reform, label="State 1 — Intermediate (ΔP only)")

print(f"\n  Utility and labour changes (State 1 vs State 0, effect of ΔP alone):")
print(f"  {'Type':>5}  {'U_base':>12}  {'U_uncomp':>12}  {'DU':>10}  "
      f"{'L_M_base':>9}  {'L_M_unc':>9}  {'DL_M':>8}")
for i in range(N):
    dU = res_uncomp[i]['U'] - U_base[i]
    dL = res_uncomp[i]['L'] - L_base[i]
    print(f"  {i+1:>5}  {U_base[i]:>12.4f}  {res_uncomp[i]['U']:>12.4f}  {dU:>+10.4f}  "
          f"{L_base[i]:>9.2f}  {res_uncomp[i]['L']:>9.2f}  {dL:>+8.2f}")

print(f"\n  STATE 2 — Pointwise schedule construction T̃(y) (U-equating, fixed L_M)")
# STATE 2 / DISTRIBUTION-NEUTRAL VAT REFORM
# CONSTRUCTION OF DN SCHEDULE
_y_base_arr = np.array([res_base[i]['y'] for i in range(N)])
_w_arr      = np.array([wages[i]          for i in range(N)])
_order      = np.argsort(_y_base_arr)
_y_sorted   = _y_base_arr[_order]
_w_sorted   = _w_arr[_order] # Builds a sorted income-wage map from baseline equillibrium for interpolation.


# Recovers the underlying implied wage at any gross income y: flat interpolation at the bottom, linear extrapolation at the top and in between
# Reports the implied labor at every single possible income y.
def w_of_y(y_g):
    if y_g <= _y_sorted[0]:
        return float(_w_sorted[0])
    if y_g >= _y_sorted[-1]:
        slope = (_w_sorted[-1] - _w_sorted[-2]) / (_y_sorted[-1] - _y_sorted[-2])
        return float(_w_sorted[-1] + slope * (y_g - _y_sorted[-1]))
    return float(np.interp(y_g, _y_sorted, _w_sorted))

#full utility at disposable income c holding L_M fixed at status quo level.
def _U_at_fixed_L(prices, c, L_fixed):
    """Full utility U(c, L_M = L_fixed) with Becker home labour + IRS at given
    commodity prices.  Returns None if infeasible."""
    info = consumer_at_cash(*prices, c)
    if info is None:
        return None
    x1_v = info['x1']
    L_H  = gamma * x1_v**theta if x1_v > 0 else 0.0
    L_total = L_fixed + L_H
    v_L = L_total**(1 + 1/epsilon) / ((1 + 1/epsilon) * phi)
    return info['usub'] - v_L

# Sets up income grid from 0 to 3000. Gives reform prices and minimum disposable income (for subsistence coverage)
y_grid       = np.linspace(0.0, 3000.0, 501)
T_tilde_vals = np.zeros_like(y_grid)
_p_reform    = np.array([1+t_reform[0], 1+t_reform[1], 1+t_reform[2]])
_c_min       = _p_reform @ k + 1e-3
fallback_count = 0

# for each grid point y, recover wage and hours, compute baseline disposible income and target utility (utility at y_g under state 0)
for j, y_g in enumerate(y_grid):
    w_g     = w_of_y(y_g)
    L_fixed = y_g / w_g if w_g > 0 else 0.0
    T_y_ref = T_ref(y_g)
    c_base  = y_g - T_y_ref

    U_tgt = _U_at_fixed_L(tuple(t_base), c_base, L_fixed)
    if U_tgt is None:
        T_tilde_vals[j] = T_y_ref
        fallback_count += 1
        continue

    def _diff(c): #root-find disposable income c such that utility under reform prices equals state 0 target.
        U_r = _U_at_fixed_L(tuple(t_reform), c, L_fixed)
        if U_r is None:
            return -1e10
        return U_r - U_tgt

    c_lo = _c_min
    c_hi = max(c_base * 5.0, 5000.0)
    try:
        c_tilde = brentq(_diff, c_lo, c_hi, xtol=1e-9, rtol=1e-9, maxiter=200)
    except Exception:
        c_tilde = c_base
        fallback_count += 1
    T_tilde_vals[j] = y_g - c_tilde

# linear interpolation of the constructed schedule for use in the household problem
def T_tilde(y):
    """Linear interpolation of the constructed schedule."""
    return float(np.interp(y, y_grid, T_tilde_vals))

# STATE 2 EQUILLIBRIUM SOLVE
# RESOLVE household at reform prices under the new schedule (Now letting L_M vary again)
res_state2 = [optimize_household(wages[i], tuple(t_reform), T_tilde) for i in range(N)]

rev_comp = sum(
    pop[i] * (res_state2[i]['T']
              + t_reform[0] * res_state2[i]['x1']
              + t_reform[1] * res_state2[i]['x2']
              + t_reform[2] * res_state2[i]['x3'])
    for i in range(N)
)
print(f"  T̃ grid: {len(y_grid)} points; brentq fallback fires: {fallback_count}")

print(f"  T̃(y) sample (interpolated from {len(y_grid)} grid points):")
print(f"  {'y':>8}  {'T(y)':>10}  {'T̃(y)':>10}  {'ΔT̃-T':>9}")
for y_sample in [0, 100, 250, 500, 850, 1000, 1500, 2000]:
    T_r = T_ref(y_sample)
    T_t = T_tilde(y_sample)
    print(f"  {y_sample:>8.0f}  {T_r:>+10.2f}  {T_t:>+10.2f}  {T_t-T_r:>+9.2f}")

print(f"\n  Government revenue R¹: {rev_comp:+,.2f} tDKK per person")
print(f"  {'Type':>5}  {'w':>6}  {'L_M_base':>9}  {'L_M_st2':>9}  {'ΔL_M':>9}  "
      f"{'Δusub':>10}  {'ΔU':>10}  {'U OK?':>6}")
for i in range(N):
    r2     = res_state2[i]
    dL     = r2['L']    - L_base[i]
    dusub  = r2['usub'] - usub_base[i]
    dU     = r2['U']    - U_base[i]
    U_ok   = abs(dU) < 1e-3
    print(f"  {i+1:>5}  {wages[i]:>6.2f}  {L_base[i]:>9.2f}  {r2['L']:>9.2f}  {dL:>+9.2f}  "
          f"{dusub:>+10.4f}  {dU:>+10.4f}  {'YES' if U_ok else 'NO':>6}")
print_shares(res_state2, *t_reform, label="State 2 — schedule construction")

# EFFICIENCY TEST
# S/Delta R = R^2 -R^0
surplus = rev_comp - rev_base
gross_income_pp = sum(pop[i] * res_base[i]['y'] for i in range(N))
surplus_pct = 100.0 * surplus / gross_income_pp if gross_income_pp != 0 else float('nan')

print(f"\n  EFFICIENCY TEST — S = R¹ − R⁰")
print(f"  R⁰ = {rev_base:+,.4f}  tDKK per person  (State 0, baseline)")
print(f"  R¹ = {rev_comp:+,.4f}  tDKK per person  (State 2, schedule construction)")
print(f"  S  = {surplus:+,.4f}  tDKK per person   "
      f"= {surplus_pct:+.4f}% of gross income per person ({gross_income_pp:,.2f} tDKK)")
if surplus > 0:
    print(f"  → S > 0: T̃ raises net revenue while preserving full-utility equivalence at every y.")
else:
    print(f"  → S < 0: T̃ loses net revenue. Reverse reform dominates.")

rev_base_com = sum(
    pop[i] * (t_base[0] * res_base[i]['x1']
              + t_base[1] * res_base[i]['x2']
              + t_base[2] * res_base[i]['x3'])
    for i in range(N)
)
rev_uncomp_com = sum(
    pop[i] * (t_reform[0] * res_uncomp[i]['x1']
              + t_reform[1] * res_uncomp[i]['x2']
              + t_reform[2] * res_uncomp[i]['x3'])
    for i in range(N)
)
rev_comp_com = sum(
    pop[i] * (t_reform[0] * res_state2[i]['x1']
              + t_reform[1] * res_state2[i]['x2']
              + t_reform[2] * res_state2[i]['x3'])
    for i in range(N)
)

print(f"\n  Revenue summary:")
print(f"    R⁰  (State 0)  : {rev_base:>12,.4f} tDKK per person")
print(f"    R   (State 1)  : {rev_uncomp:>12,.4f} tDKK per person  (ΔP only; ΔR = {rev_uncomp-rev_base:+,.4f})")
print(f"    R¹  (State 2)  : {rev_comp:>12,.4f} tDKK per person  (S = {surplus:+,.4f}, {surplus_pct:+.4f}% of gross y)")

print(f"\n  Commodity-tax revenue (per person, pop-weighted Σ τ_j · x_j):")
print(f"    State 0  : {rev_base_com:>12,.4f} tDKK   ({100*rev_base_com/rev_base:>5.2f}% of R⁰)")
print(f"    State 1  : {rev_uncomp_com:>12,.4f} tDKK   ({100*rev_uncomp_com/rev_uncomp:>5.2f}% of R)   (ΔP only; Δ = {rev_uncomp_com-rev_base_com:+,.4f})")
print(f"    State 2  : {rev_comp_com:>12,.4f} tDKK   ({100*rev_comp_com/rev_comp:>5.2f}% of R¹)  (Δ vs State 0 = {rev_comp_com-rev_base_com:+,.4f})")

# sanity check that labour responses are small (they should be zero under weak seperabiltiy, and near zero under seperability)
_dL_max = max(abs(res_state2[i]['L'] - L_base[i]) for i in range(N))
print(f"  Max |ΔL_M| (market labour response, NOT imposed): {_dL_max:.2e} hours/year")
print(f"  Population-weighted ΔL_M: "
      f"{sum(pop[i]*(res_state2[i]['L']-L_base[i]) for i in range(N)):+.4f} hours/year")


# below is a wrapper that recomputes the entire state 0 -> DN-T -> State 2 pipeline at different gamma, then restores to the original.
# The try/finally ensures that gamma is reset even on error. Used for gamma = 0 sanity check.
def _compute_S_at_gamma(gamma_val, t_reform_local=None):
    """Run the full State 0 / T̃ build / State 2 pipeline at a given γ
    and return the surplus S = R¹ − R⁰."""
    global gamma
    _g_save = gamma
    gamma = gamma_val
    if t_reform_local is None:
        t_reform_local = tuple(t_reform)
    try:
        rb = [optimize_household(wages[i], tuple(t_base), T_ref) for i in range(N)]
        Rb = sum(pop[i] * (rb[i]['T']
                            + t_base[0]*rb[i]['x1']
                            + t_base[1]*rb[i]['x2']
                            + t_base[2]*rb[i]['x3'])
                 for i in range(N))

        ya = np.array([rb[i]['y'] for i in range(N)])
        wa = np.array([wages[i]   for i in range(N)])
        order = np.argsort(ya)
        ys = ya[order]; ws = wa[order]
        def _w_of_y(y):
            if y <= ys[0]:  return float(ws[0])
            if y >= ys[-1]: return float(ws[-1])
            return float(np.interp(y, ys, ws))

        def _UfL(prices, c, L_fixed):
            info = consumer_at_cash(*prices, c)
            if info is None: return None
            x1_v = info['x1']
            L_H  = gamma * x1_v**theta if x1_v > 0 else 0.0
            L_total = L_fixed + L_H
            v_L = L_total**(1+1/epsilon) / ((1+1/epsilon) * phi)
            return info['usub'] - v_L

        ygrid = np.linspace(0.0, 3000.0, 501)
        Tt    = np.zeros_like(ygrid)
        pr    = np.array([1+t_reform_local[0], 1+t_reform_local[1], 1+t_reform_local[2]])
        cmin  = pr @ k + 1e-3
        for j, yg in enumerate(ygrid):
            wg = _w_of_y(yg)
            Lf = yg / wg if wg > 0 else 0.0
            Tref_v = T_ref(yg)
            cb     = yg - Tref_v
            Utgt   = _UfL(tuple(t_base), cb, Lf)
            if Utgt is None:
                Tt[j] = Tref_v;  continue
            def _d(c):
                Ur = _UfL(tuple(t_reform_local), c, Lf)
                return -1e10 if Ur is None else (Ur - Utgt)
            try:
                ct = brentq(_d, cmin, max(cb*5.0, 5000.0),
                            xtol=1e-9, rtol=1e-9, maxiter=200)
            except Exception:
                ct = cb
            Tt[j] = yg - ct

        def _Tt_fn(y): return float(np.interp(y, ygrid, Tt))

        rc = [optimize_household(wages[i], tuple(t_reform_local), _Tt_fn) for i in range(N)]
        Rc = sum(pop[i] * (rc[i]['T']
                            + t_reform_local[0]*rc[i]['x1']
                            + t_reform_local[1]*rc[i]['x2']
                            + t_reform_local[2]*rc[i]['x3'])
                 for i in range(N))
        return Rc - Rb
    finally:
        gamma = _g_save

# Runs a sanity check at gamma = 0 (home production off / weak seperability). 
T_REFORM_SANITY = (0.0, 0.25, 0.25)
S_REF_SEPARABLE = None
S_g0 = _compute_S_at_gamma(0.0, t_reform_local=T_REFORM_SANITY)
print(f"\n  SANITY CHECK at γ = 0 with reform = {T_REFORM_SANITY}  (θ = {theta} irrelevant here):")
if S_REF_SEPARABLE is not None:
    diff = S_g0 - S_REF_SEPARABLE
    print(f"γ=0 sanity: S = {S_g0:+.6f}, reference = {S_REF_SEPARABLE:+.6f}, diff = {diff:+.2e}")
    assert abs(diff) < 1e-3, (
        f"At γ=0 the U-equating construction must match the V-equating reference; "
        f"got S = {S_g0:+.6f}, expected ~{S_REF_SEPARABLE:+.4f}."
    )
    print(f"  PASS — γ = 0 reproduces separable reference (within 1e-3).")
else:
    print(f"  γ=0 result: S = {S_g0:+.6f}  (set S_REF_SEPARABLE to enable assertion)")
print(f"\n  Note: module-level t_reform = {tuple(t_reform)} is used for the main run "
      f"above; the sanity check uses the canonical reform tuple regardless.")

# CABLES NEEDED FOR EFFECTIVE CALIBRATION
print(f"\n  SHARE FIT AT BASELINE  (implied vs observed)")
print(f"  {'Type':>5}  {'s1_obs':>8}  {'s1_imp':>8}  {'s2_obs':>8}  {'s2_imp':>8}  {'s3_obs':>8}  {'s3_imp':>8}")
_calib = [optimize_household(wages[i], tuple(t_base), T_ref) for i in range(N)]
for i in range(N):
    r  = _calib[i]
    pb = 1 + t_base
    M  = pb[0]*r['x1'] + pb[1]*r['x2'] + pb[2]*r['x3']
    s  = np.array([pb[j]*r[f'x{j+1}'] for j in range(3)]) / M
    so = _obs_shares[i]
    print(f"  {i+1:>5}  {so[0]:>8.2%}  {s[0]:>8.2%}  {so[1]:>8.2%}  {s[1]:>8.2%}  {so[2]:>8.2%}  {s[2]:>8.2%}")
print(f"  beta={list(beta)}  alpha={alpha}  k={list(k)}  sigma_n={sigma_n}  sigma_o={sigma_o}  gamma={gamma}  theta={theta}")

_L_baseline = [_calib[i]['L'] for i in range(N)]
_L_avg = sum(pop[i] * _L_baseline[i] for i in range(N))

print(f"\n  CALIBRATION CHECK — target vs model gross income")
print(f"  {'Type':>5}  {'w':>10}  {'L_M_model':>10}  {'y_target':>10}  {'y_model':>10}  {'diff %':>8}")
for i in range(N):
    y_model = wages[i] * _L_baseline[i]
    y_target = _annual_income[i]
    pct = 100 * (y_model - y_target) / y_target
    print(f"  {i+1:>5}  {wages[i]:>10.5f}  {_L_baseline[i]:>10.2f}  "
          f"{y_target:>10.2f}  {y_model:>10.2f}  {pct:>+7.1f}%")

print(f"\n  CALIBRATION CHECK — labour supply at baseline (L_M = market hours, L_H = home hours)")
print(f"  {'Type':>5}  {'w (tDKK/hr)':>12}  {'MTR':>6}  {'L_M (hours)':>12}  {'L_H (hours)':>12}")
_L_H_baseline = [
    gamma * _calib[i]['x1']**theta if _calib[i]['x1'] > 0 else 0.0
    for i in range(N)
]
for i in range(N):
    y_model = wages[i] * _L_baseline[i]
    print(f"  {i+1:>5}  {wages[i]:>12.5f}  {MTR_of_y(y_model):>6.1%}  "
          f"{_L_baseline[i]:>12.2f}  {_L_H_baseline[i]:>12.2f}")
_L_H_avg = sum(pop[i] * _L_H_baseline[i] for i in range(N))
print(f"  Population-weighted average L_M: {_L_avg:.2f} hours/year  (target: {HOURS_YEAR:.0f})")
print(f"  Population-weighted average L_H: {_L_H_avg:.2f} hours/year")
