"""
Redraws manuscript Figures 1-6 from the outputs of full_pipeline.py.
Usage: python make_figures.py punjab_cgwb_dtw_wellwise_parsed.csv results_folder figures_folder
Numbering follows the manuscript: Fig 1 coverage/distribution, Fig 2 district change, Fig 3 gauged thinning,
Fig 4 ungauged thinning, Fig 5 (a) value of history (b) district RMSE, Fig 6 interval calibration.
"""
import sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV, RES, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 9,
                     "axes.labelsize": 8, "legend.fontsize": 7, "figure.dpi": 120, "savefig.dpi": 300, "font.family": "DejaVu Sans"})
C = {"pers": "#7f7f7f", "sn": "#8c564b", "sn_dist": "#bcbd22", "shift": "#1f77b4", "ridge": "#2ca02c", "rf": "#9467bd", "gb": "#d62728",
     "global": "#7f7f7f", "district": "#1f77b4", "shrunk": "#2ca02c", "ens": "#d62728", "conf": "#1f4e9c"}
NAMES = {"pers": "Persistence", "sn": "Seasonal-naive", "sn_dist": "Seasonal-naive + district shift", "shift": "District-shift",
         "ridge": "Ridge", "rf": "Random forest", "gb": "Boosting", "global": "Global mean", "district": "District mean",
         "shrunk": "Shrunk district mean"}
def rd(n, **k): return pd.read_csv(os.path.join(RES, n + ".csv"), **k)
def nice(s): return {"SAS NAGAR": "SAS Nagar", "SBS NAGAR": "SBS Nagar", "TARAN TARAN": "Taran Taran"}.get(s, s.title())
def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), bbox_inches="tight"); plt.close(fig); print("saved", name)

# ------------------------------------------------------------------ Fig 1
raw = pd.read_csv(CSV); R = [c for c in raw.columns if c[:2] == "20"]
raw["range"] = raw[R].max(axis=1) - raw[R].min(axis=1); ok = raw[raw.range <= 12]
LAB = {"2019-06": "Jun 19", "2019-08": "Aug 19", "2019-10": "Oct 19", "2020-01": "Jan 20", "2022-06": "Jun 22", "2022-08": "Aug 22",
       "2022-11": "Nov 22", "2023-01": "Jan 23", "2023-06": "Jun 23", "2023-08": "Aug 23", "2023-11": "Nov 23", "2024-01": "Jan 24"}
cols3 = ["#4c78a8"] * 4 + ["#f58518"] * 4 + ["#54a24b"] * 4
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0))
ax[0].bar(range(12), [ok[c].notna().sum() for c in R], color=cols3)
ax[0].set_xticks(range(12)); ax[0].set_xticklabels([LAB[c] for c in R], rotation=60, ha="right")
ax[0].set_ylabel("Wells with a reading"); ax[0].set_title("(a) Network coverage per round")
from matplotlib.patches import Patch
ax[0].legend(handles=[Patch(color="#4c78a8", label="2019–20"), Patch(color="#f58518", label="2022–23"), Patch(color="#54a24b", label="2023–24")], frameon=False, loc="upper right")
ax[1].boxplot([ok[c].dropna() for c in R], flierprops=dict(marker="o", markersize=2), medianprops=dict(color="k"))
ax[1].set_xticklabels([LAB[c] for c in R], rotation=60, ha="right"); ax[1].set_ylabel("Depth to water level (m bgl)")
ax[1].set_title("(b) Distribution of readings"); fig.tight_layout(); save(fig, "Fig1_coverage_distribution")

# ------------------------------------------------------------------ Fig 2
t3 = rd("table3_district_change").sort_values("median")
fig, ax = plt.subplots(figsize=(5.2, 4.0))
ax.barh([f"{nice(d)} ({int(n)})" for d, n in zip(t3.district, t3.n)], t3["median"], color=np.where(t3["median"] > 0, "#c0392b", "#2e6da4"))
ax.axvline(0, color="k", lw=0.6); ax.set_xlabel("Median change in June depth to water level, 2019 to 2023 (m)")
fig.tight_layout(); save(fig, "Fig2_district_change")

# ------------------------------------------------------------------ Fig 3 / Fig 4 (thinning with bands)
def thin_panel(ax, df, models, band_models, title):
    g = df.groupby(["f", "model"]).rmse.agg(mean="mean", lo=lambda s: s.quantile(.025), hi=lambda s: s.quantile(.975)).reset_index()
    for m in models:
        d = g[g.model == m].sort_values("f", ascending=False); x = d.f * 100
        flat = m in ("pers", "sn", "global")
        ax.plot(x, d["mean"], marker="o", ms=3, lw=1.4, ls="--" if flat else "-", color=C[m], label=NAMES[m])
        if m in band_models: ax.fill_between(x, d.lo, d.hi, color=C[m], alpha=0.12, lw=0)
    ax.set_xlim(104, 6); ax.set_xticks([100, 75, 50, 25, 10]); ax.set_xlabel("Training wells retained (%)"); ax.set_ylabel("RMSE on withheld wells (m)"); ax.set_title(title)
for fig_name, prefix, models, bands, tt, ylim in [
        ("Fig3_gauged_thinning", "splits_scenarioB_", ["pers", "sn", "shift", "ridge", "rf", "gb"], ["ridge", "gb"], ("(a) Target: Nov 2023", "(b) Target: Jun 2023"), None),
        ("Fig4_ungauged_thinning", "splits_scenarioA_", ["global", "district", "shrunk", "gb"], ["district", "gb"], ("(a) Target: Nov 2023", "(b) Target: Jun 2023"), None)]:
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=False)
    for a, tgt, t in zip(ax, ["2023-11", "2023-06"], tt):
        thin_panel(a, pd.read_csv(os.path.join(RES, f"{prefix}{tgt}.csv")), models, bands, t)
    h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, ncol=len(models), loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1)); save(fig, fig_name)

# ------------------------------------------------------------------ Fig 5 (history + district RMSE)
t7 = rd("table7_value_of_history"); fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.6), gridspec_kw={"width_ratios": [1, 1.1]})
labs = ["Jun-23\nonly", "+ Nov-22", "+ Jun-22", "+ 2019\nrounds", "Prior years\nonly*"]; x = np.arange(len(t7)); w = 0.2
for i, (c, col, nm) in enumerate([("Persistence", C["pers"], "Persistence"), ("District-shift", C["shift"], "District-shift"), ("Ridge", C["ridge"], "Ridge"), ("Boosting", C["gb"], "Boosting")]):
    ax[0].bar(x + (i - 1.5) * w, t7[c], w, color=col, label=nm)
ax[0].set_xticks(x); ax[0].set_xticklabels(labs, fontsize=7); ax[0].set_ylabel("RMSE, Nov 2023 (m)"); ax[0].set_ylim(1.0, 2.75)
ax[0].set_title("(a) Value of a well's own history"); ax[0].legend(frameon=False, ncol=2, loc="upper right"); ax[0].text(0.0, -0.22, "*different set of wells", transform=ax[0].transAxes, fontsize=6.5)
dd = rd("fig_data_district_rmse_ungauged"); dd = dd[dd.target == "2023-11"].pivot(index="district", columns="f", values="rmse").sort_values(1.0)
y = np.arange(len(dd)); ax[1].barh(y + 0.2, dd[1.0], 0.38, color="#1f77b4", label="100% retained"); ax[1].barh(y - 0.2, dd[0.25], 0.38, color="#f39c12", label="25% retained")
ax[1].set_yticks(y); ax[1].set_yticklabels([nice(s) for s in dd.index], fontsize=6.5); ax[1].set_xlabel("District-mean RMSE, Nov 2023 (m)")
ax[1].set_title("(b) Ungauged wells by district"); ax[1].legend(frameon=False, loc="lower right"); fig.tight_layout(); save(fig, "Fig5_history_and_district_rmse")

# ------------------------------------------------------------------ Fig 6 (calibration)
U = pd.read_csv(os.path.join(RES, "raw_uncertainty.csv")); cov = U[U.level != "crps"].copy(); cov["level"] = cov.level.astype(float)
fig, ax = plt.subplots(1, 3, figsize=(7.6, 2.9))
for tgt, ls, lab in [("2023-11", "-", "Nov 2023"), ("2023-06", "--", "Jun 2023")]:
    s = cov[(cov.target == tgt) & (cov.level == 0.9)].groupby("f")[["picp_ens", "picp_conf", "width_ens", "width_conf"]].mean().sort_index(ascending=False)
    ax[0].plot(s.index * 100, s.picp_ens, ls, color=C["ens"], marker="o", ms=3, label=f"Ensemble, {lab}")
    ax[0].plot(s.index * 100, s.picp_conf, ls, color=C["conf"], marker="o", ms=3, label=f"Conformal, {lab}")
    ax[2].plot(s.index * 100, s.width_ens, ls, color=C["ens"], marker="o", ms=3); ax[2].plot(s.index * 100, s.width_conf, ls, color=C["conf"], marker="o", ms=3)
ax[0].axhline(0.9, color="k", ls=":", lw=0.8); ax[0].set_ylim(0.3, 1.0); ax[0].set_ylabel("Coverage of nominal 90% interval"); ax[0].set_title("(a) Coverage")
r = cov[(cov.target == "2023-11") & (cov.f == 0.25)].groupby("level")[["picp_ens", "picp_conf"]].mean()
ax[1].plot([0.4, 1], [0.4, 1], "k:", lw=0.8); ax[1].plot(r.index, r.picp_ens, color=C["ens"], marker="o", ms=3); ax[1].plot(r.index, r.picp_conf, color=C["conf"], marker="o", ms=3)
ax[1].set_xlabel("Nominal coverage"); ax[1].set_ylabel("Empirical coverage"); ax[1].set_title("(b) Reliability, 25% retained"); ax[1].set_xlim(0.45, 1); ax[1].set_ylim(0.1, 1)
ax[2].set_ylabel("Mean 90% interval width (m)"); ax[2].set_title("(c) Width")
for a in (ax[0], ax[2]): a.set_xlim(104, 6); a.set_xticks([100, 50, 25, 10]); a.set_xlabel("Training wells retained (%)")
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, ncol=4, loc="lower center", frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.07, 1, 1)); save(fig, "Fig6_interval_calibration")
