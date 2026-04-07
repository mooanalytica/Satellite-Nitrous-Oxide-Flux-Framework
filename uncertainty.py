"""
08_uncertainty.py — Parameter sensitivity (tornado) & Monte Carlo UQ.

Functions:
    sensitivity(flat_arrays, baseline_Gg) -> DataFrame
    monte_carlo(flat_arrays, baseline_Gg, n_mc=1000, sample_n=500_000)
        -> (mc_totals_array, summary_dict)
"""

import os
import time

import numpy as np
import pandas as pd

from src.config import N_EXCRETION, MMS_FRAC, ANIMAL_COLS, OUT_DIR
from src.grid import PIXEL_HA

PARAM_RANGES = {
    "Base EF (IPCC +/-50%)":       ("base_ef",    0.005, 0.015),
    "Drainage factor (Very Poor)": ("drain_vp",   1.5,   2.5),
    "Clay texture slope":          ("clay_slope", 0.010, 0.020),
    "SMAP wet SM factor":          ("sm_wet",     1.3,   2.0),
    "SMAP dry SM factor":          ("sm_dry",     0.5,   0.8),
    "Temperature warm factor":     ("temp_warm",  1.1,   1.5),
    "Freeze-thaw coefficient":     ("ft_coef",    0.002, 0.006),
    "Organic carbon slope":        ("oc_slope",   0.05,  0.15),
    "NDVI uptake (dense)":         ("ndvi_up",    0.75,  0.95),
    "Leaching fraction (wet)":     ("leach_wet",  0.20,  0.40),
    "MMS fraction (dairy)":        ("mms_dairy",  0.50,  0.80),
    "N excretion rate (dairy)":    ("nex_dairy",  100,   140),
}

MC_PARAMS = {
    "base_ef":    (0.010, 0.003, "lognormal"),
    "drain_vp":   (2.00,  0.25,  "normal"),
    "clay_slope": (0.015, 0.003, "normal"),
    "sm_wet":     (1.60,  0.15,  "normal"),
    "sm_dry":     (0.70,  0.10,  "normal"),
    "temp_warm":  (1.30,  0.10,  "normal"),
    "ft_coef":    (0.004, 0.001, "normal"),
    "oc_slope":   (0.10,  0.03,  "normal"),
    "ndvi_up":    (0.85,  0.05,  "normal"),
    "leach_wet":  (0.30,  0.06,  "normal"),
    "mms_dairy":  (0.65,  0.08,  "normal"),
    "nex_dairy":  (120.0, 15.0,  "normal"),
}


def _compute_flat(p, v, fa):
    dv, dvp, dpp = fa["dv"], fa["dvp"], fa["dpp"]
    cl, sm, tc, ft, oc, nd, ca, pr = (
        fa["cl"], fa["sm"], fa["tc"], fa["ft"], fa["oc"], fa["nd"], fa["ca"], fa["pr"])
    an, n = fa["an"], len(cl)

    d_f  = np.where(dvp, max(0.5, v if p == "drain_vp" else 2.0), dv)
    cs   = v if p == "clay_slope" else 0.015
    t_f  = 1.0 + np.maximum(0, cl - 20) * cs
    sw   = v if p == "sm_wet"   else 1.6
    sd   = v if p == "sm_dry"   else 0.7
    sm_f = np.where(sm > 0.35, sw, np.where(sm > 0.25, 1.2, np.where(sm > 0.15, 1.0, sd)))
    tw   = v if p == "temp_warm" else 1.3
    tp_f = np.where(tc > 10, tw, np.where(tc > 5, 1.1, np.where(tc > 0, 1.0, 0.8)))
    fc   = v if p == "ft_coef"  else 0.004
    ft_f = 1.0 + ft * max(0, fc)
    ocs  = v if p == "oc_slope" else 0.10
    oc_f = 1.0 + np.maximum(0, oc - 2.0) * max(0, ocs)
    bef  = v if p == "base_ef"  else 0.01
    ef   = np.clip(bef * d_f * t_f * sm_f * tp_f * ft_f * oc_f, 0.001, 0.06)
    nu   = v if p == "ndvi_up"  else 0.85
    n_up = np.where(nd > 0.7, nu, np.where(nd > 0.5, 0.92, np.where(nd > 0.3, 1.0, 1.05)))
    md   = v if p == "mms_dairy" else MMS_FRAC["dairy"]
    ndv  = v if p == "nex_dairy" else N_EXCRETION["dairy"]

    N_app = np.zeros(n); N_pas = np.zeros(n)
    for animal, nex_b in N_EXCRETION.items():
        nex   = ndv  if animal == "dairy" else nex_b
        mf_b  = md   if animal == "dairy" else MMS_FRAC[animal]
        n_ex  = an[animal] * nex
        mf    = np.clip(mf_b * (0.5 + 0.5 * ca), 0, 1)
        N_app += n_ex * mf; N_pas += n_ex * (1 - mf)

    eff_N = N_app * n_up
    N2O_d = eff_N * ef * (44/28) + N_pas * 0.02 * (44/28)
    lw    = v if p == "leach_wet" else 0.30
    fl    = np.where(pr > 800, lw, np.where(pr > 600, 0.24, np.where(pr > 400, 0.15, 0.08)))
    dm    = np.where(dpp, 1.5, 1.0)
    N2O_i = eff_N * 0.20 * 0.01 * (44/28) + eff_N * fl * 0.0075 * dm * (44/28)
    return (N2O_d + N2O_i).sum() * PIXEL_HA / 1e6


def sensitivity(flat_arrays, baseline_Gg) -> pd.DataFrame:
    rows = []
    for label, (param, lo, hi) in PARAM_RANGES.items():
        n2o_lo = _compute_flat(param, lo, flat_arrays)
        n2o_hi = _compute_flat(param, hi, flat_arrays)
        rows.append({
            "Parameter": label,
            "Low_Gg":    n2o_lo,
            "High_Gg":   n2o_hi,
            "Range_Gg":  abs(n2o_hi - n2o_lo),
            "Range_pct": abs(n2o_hi - n2o_lo) / baseline_Gg * 100,
        })
    df = pd.DataFrame(rows).sort_values("Range_Gg", ascending=False)
    df.to_csv(os.path.join(OUT_DIR, "parameter_sensitivity.csv"), index=False, float_format="%.4f")
    return df


def monte_carlo(flat_arrays, baseline_Gg, n_mc=1000, sample_n=500_000):
    fa      = flat_arrays
    n       = len(fa["cl"])
    rng     = np.random.default_rng(0)
    totals  = np.zeros(n_mc)
    t0      = time.time()

    for i in range(n_mc):
        s = {}
        for p, (mu, sigma, dist) in MC_PARAMS.items():
            if dist == "lognormal":
                sl   = np.sqrt(np.log(1 + (sigma / mu) ** 2))
                ml   = np.log(mu) - 0.5 * sl ** 2
                s[p] = rng.lognormal(ml, sl)
            else:
                s[p] = rng.normal(mu, sigma)

        d_f  = np.where(fa["dvp"], max(0.5, s["drain_vp"]), fa["dv"])
        t_f  = 1.0 + np.maximum(0, fa["cl"] - 20) * max(0, s["clay_slope"])
        sw   = max(1.0, s["sm_wet"])
        sd   = max(0.3, s["sm_dry"])
        sm_f = np.where(fa["sm"] > 0.35, sw,
               np.where(fa["sm"] > 0.25, 1.2,
               np.where(fa["sm"] > 0.15, 1.0, sd)))
        tw   = max(1.0, s["temp_warm"])
        tp_f = np.where(fa["tc"] > 10, tw,
               np.where(fa["tc"] > 5,  1.1,
               np.where(fa["tc"] > 0,  1.0, 0.8)))
        ft_f = 1.0 + fa["ft"] * max(0, s["ft_coef"])
        oc_f = 1.0 + np.maximum(0, fa["oc"] - 2.0) * max(0, s["oc_slope"])
        ef   = np.clip(s["base_ef"] * d_f * t_f * sm_f * tp_f * ft_f * oc_f, 0.001, 0.06)
        nu   = np.clip(s["ndvi_up"], 0.70, 1.0)
        n_up = np.where(fa["nd"] > 0.7, nu,
               np.where(fa["nd"] > 0.5, 0.92,
               np.where(fa["nd"] > 0.3, 1.0, 1.05)))
        md   = np.clip(s["mms_dairy"], 0.3, 0.95)
        ndv  = max(80, s["nex_dairy"])
        N_app = np.zeros(n); N_pas = np.zeros(n)
        for animal, nex_b in N_EXCRETION.items():
            nex  = ndv if animal == "dairy" else nex_b
            mf_b = md  if animal == "dairy" else MMS_FRAC[animal]
            n_ex = fa["an"][animal] * nex
            mf   = np.clip(mf_b * (0.5 + 0.5 * fa["ca"]), 0, 1)
            N_app += n_ex * mf; N_pas += n_ex * (1 - mf)
        eff_N  = N_app * n_up
        N2O_d  = eff_N * ef * (44/28) + N_pas * 0.02 * (44/28)
        lw     = np.clip(s["leach_wet"], 0.10, 0.50)
        fl     = np.where(fa["pr"] > 800, lw,
                 np.where(fa["pr"] > 600, 0.24,
                 np.where(fa["pr"] > 400, 0.15, 0.08)))
        dm     = np.where(fa["dpp"], 1.5, 1.0)
        N2O_i  = eff_N * 0.20 * 0.01 * (44/28) + eff_N * fl * 0.0075 * dm * (44/28)
        total  = (N2O_d + N2O_i).sum() * PIXEL_HA / 1e6
        totals[i] = total if np.isfinite(total) else np.nan

    totals = totals[np.isfinite(totals)]
    summary = {
        "mean_Gg": float(totals.mean()),
        "std_Gg":  float(totals.std()),
        "cv_pct":  float(totals.std() / totals.mean() * 100),
        "p5_Gg":   float(np.percentile(totals, 5)),
        "p95_Gg":  float(np.percentile(totals, 95)),
        "elapsed_min": (time.time() - t0) / 60,
    }
    np.save(os.path.join(OUT_DIR, "mc_distribution.npy"), totals)
    pd.DataFrame([summary]).to_csv(
        os.path.join(OUT_DIR, "mc_summary.csv"), index=False, float_format="%.4f")
    return totals, summary
