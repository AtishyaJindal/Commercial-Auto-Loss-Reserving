"""
Reserve_Bootstrap.py
=====================
Turns the single-point Chain-Ladder reserve estimate into a full probability
distribution using the ODP (Over-Dispersed Poisson) bootstrap method
(England & Verrall, 2002) -- the industry-standard way actuaries quantify
reserve uncertainty.

Data: CAS Schedule P, State Farm Mut Grp, Commercial Auto.

Two sources of uncertainty are simulated:
  1. PARAMETER uncertainty -- we don't know the "true" development factors,
     only an estimate from 10 years of noisy data. We resample the model's
     own residuals to generate thousands of alternate histories consistent
     with the data, and refit Chain-Ladder to each ("bootstrapping the fit").
  2. PROCESS uncertainty -- even with the *correct* development factors,
     next year's actual claim payments are still a random draw, not a fixed
     number. We simulate that draw from a Gamma distribution calibrated to
     the data's own variance structure.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)
N_SIM = 10000
TAIL_FACTOR = 1.010  # same judgmental tail assumption as the Excel model

# ---------------------------------------------------------------- load data
paid = pd.read_csv("paid_triangle.csv", index_col=0)
paid.columns = paid.columns.astype(int)
AYS = paid.index.tolist()
N = len(AYS)  # 10 accident years / dev periods

# cumulative -> incremental triangle (upper triangle only; NaN = not yet observed)
incr = paid.copy()
for j in range(2, 11):
    incr[j] = paid[j] - paid[j - 1]
incr[1] = paid[1]
incr = incr[sorted(incr.columns)]

# ---------------------------------------------------------- 1) fit chain-ladder
def volume_weighted_factors(cum_tri):
    """Standard chain-ladder age-to-age factors from a cumulative triangle."""
    factors = {}
    for k in range(1, N):
        num = cum_tri[k + 1].dropna()
        den = cum_tri[k].loc[num.index]
        factors[k] = num.sum() / den.sum()
    return factors

factors = volume_weighted_factors(paid)
cdf = {N: TAIL_FACTOR}
for k in range(N - 1, 0, -1):
    cdf[k] = factors[k] * cdf[k + 1]

# fitted cumulative triangle: back-cast every cell (including lower/future triangle)
# using the LAST diagonal's known values run forward/backward through the factors.
fitted_cum = pd.DataFrame(index=AYS, columns=range(1, N + 1), dtype=float)
for i, ay in enumerate(AYS):
    dev_known = N - i  # last observed dev lag for this AY (1997 diagonal)
    known_val = paid.loc[ay, dev_known]
    # forward-fill future cells
    v = known_val
    fitted_cum.loc[ay, dev_known] = known_val
    for k in range(dev_known, N):
        v = v * factors[k]
        fitted_cum.loc[ay, k + 1] = v
    # back-fill earlier cells (needed to compute fitted incrementals for residuals)
    v = known_val
    for k in range(dev_known - 1, 0, -1):
        v = v / factors[k]
        fitted_cum.loc[ay, k] = v

fitted_incr = fitted_cum.copy()
for j in range(2, N + 1):
    fitted_incr[j] = fitted_cum[j] - fitted_cum[j - 1]
fitted_incr[1] = fitted_cum[1]

# ---------------------------------------------------------- 2) Pearson residuals
# scale parameter (Pearson chi-square / degrees of freedom) from the UPPER
# triangle only (the data we actually observed)
resid_list = []
n_obs = 0
ss = 0.0
for i, ay in enumerate(AYS):
    dev_known = N - i
    for j in range(1, dev_known + 1):
        actual = incr.loc[ay, j]
        fit = fitted_incr.loc[ay, j]
        if pd.notna(actual) and fit > 0:
            r = (actual - fit) / np.sqrt(abs(fit))
            resid_list.append(r)
            ss += r ** 2
            n_obs += 1

p = N * (N + 1) // 2          # number of upper-triangle cells (55)
n_params = 2 * N - 1           # AY levels + dev factors (approx, standard DOF count)
dof = max(p - n_params, 1)
scale = ss / dof
residuals = np.array(resid_list)
residuals = residuals - residuals.mean()  # centre for resampling

print(f"Number of upper-triangle cells used: {n_obs}")
print(f"Pearson scale parameter (phi): {scale:.2f}")
print(f"Residual pool size: {len(residuals)}")

# ---------------------------------------------------------- 3) bootstrap loop
sim_reserves = np.zeros(N_SIM)
sim_reserves_by_ay = np.zeros((N_SIM, N))

for s in range(N_SIM):
    # --- stage A: build a pseudo history by resampling residuals onto the fit
    pseudo_incr = fitted_incr.copy()
    for i, ay in enumerate(AYS):
        dev_known = N - i
        for j in range(1, dev_known + 1):
            r_star = np.random.choice(residuals)
            fit = fitted_incr.loc[ay, j]
            pseudo_incr.loc[ay, j] = max(fit + r_star * np.sqrt(abs(fit)), 0)

    pseudo_cum = pseudo_incr.copy()
    for j in range(2, N + 1):
        pseudo_cum[j] = pseudo_cum[j - 1] + pseudo_incr[j]
    # keep only upper triangle (mirror the real data's availability)
    pseudo_cum_upper = pseudo_cum.copy()
    for i, ay in enumerate(AYS):
        dev_known = N - i
        for j in range(dev_known + 1, N + 1):
            pseudo_cum_upper.loc[ay, j] = np.nan

    # refit chain-ladder factors on this pseudo history (parameter uncertainty)
    pf = volume_weighted_factors(pseudo_cum_upper)
    pcdf = {N: TAIL_FACTOR}
    for k in range(N - 1, 0, -1):
        pcdf[k] = pf[k] * pcdf[k + 1]

    # --- stage B: simulate the actual FUTURE process outcome (process uncertainty)
    ay_total_reserve = np.zeros(N)
    for i, ay in enumerate(AYS):
        dev_known = N - i
        latest = paid.loc[ay, dev_known]   # anchor on REAL latest diagonal
        mean_ultimate = latest * pcdf[dev_known]
        mean_future = max(mean_ultimate - latest, 0.0)
        if mean_future > 0:
            # Gamma draw: mean = mean_future, variance = scale * mean_future
            shape = mean_future / scale if scale > 0 else 1e6
            draw = np.random.gamma(shape, scale) if scale > 0 else mean_future
        else:
            draw = 0.0
        ay_total_reserve[i] = draw
    sim_reserves_by_ay[s, :] = ay_total_reserve
    sim_reserves[s] = ay_total_reserve.sum()

# ---------------------------------------------------------- 4) summarize
cl_point_estimate = 433215.01  # from the Excel Chain-Ladder model, for reference
mean_r = sim_reserves.mean()
std_r = sim_reserves.std()
pctiles = {p: np.percentile(sim_reserves, p) for p in [5, 25, 50, 75, 90, 95, 99]}
cv = std_r / mean_r

print("\n" + "=" * 60)
print("BOOTSTRAP RESERVE DISTRIBUTION SUMMARY (10,000 simulations)")
print("=" * 60)
print(f"Chain-Ladder point estimate (Excel model): ${cl_point_estimate:,.0f}  (000s)")
print(f"Bootstrap mean reserve:                    ${mean_r:,.0f}  (000s)")
print(f"Bootstrap std deviation:                   ${std_r:,.0f}  (000s)")
print(f"Coefficient of variation (CV):              {cv:.1%}")
print()
for p, v in pctiles.items():
    print(f"  {p:>3}th percentile: ${v:,.0f}  (000s)")
print()
print(f"90% confidence interval:  ${pctiles[5]:,.0f}  --  ${pctiles[95]:,.0f}  (000s)")
print(f"Implied 95% VaR-style capital margin over the mean: ${pctiles[95]-mean_r:,.0f}  (000s)")

# ---------------------------------------------------------- 5) save outputs
out = pd.DataFrame({"simulated_total_reserve": sim_reserves})
out.to_csv("bootstrap_results.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(sim_reserves, bins=80, color="#1F3864", alpha=0.85, edgecolor="white")
ax.axvline(mean_r, color="#BF9000", linewidth=2.5, label=f"Bootstrap mean: ${mean_r:,.0f}k")
ax.axvline(cl_point_estimate, color="#C00000", linewidth=2.5, linestyle="--",
           label=f"Chain-Ladder point estimate: ${cl_point_estimate:,.0f}k")
ax.axvline(pctiles[95], color="#548235", linewidth=2, linestyle=":",
           label=f"95th percentile: ${pctiles[95]:,.0f}k")
ax.set_title("Simulated Distribution of Total Unpaid Loss Reserve\nODP Bootstrap, 10,000 iterations -- State Farm Commercial Auto",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Total Reserve ($000s)")
ax.set_ylabel("Number of Simulations")
ax.legend(loc="upper right", fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("reserve_distribution.png", dpi=150)
print("\nSaved: bootstrap_results.csv, reserve_distribution.png")
