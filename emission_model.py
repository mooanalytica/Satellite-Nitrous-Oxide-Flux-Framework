"""
06_emission_model.py — Per-pixel N₂O emission model.

Unit chain:
    N_app  [kg N pixel⁻¹ yr⁻¹]  = heads × N_excretion [kg N head⁻¹ yr⁻¹]
    N2O    [kg N2O pixel⁻¹]     = N_app × EF × (44/28)
    density [kg N2O ha⁻¹]       = N2O / PIXEL_HA  (25 ha, constant)

Main entry point:
    run(covs, livestock_grids, ag_mask, drain_f, drain_poor,
        clay_grid, oc_grid, crop_ann)
    -> dict with keys: EF, N2O_direct, N2O_indirect, N2O_total, N2O_ipcc,
                       N2O_density, EF_departure_pct
"""

import numpy as np

from src.config import N_EXCRETION, MMS_FRAC, ANIMAL_COLS
from src.grid import PIXEL_HA, NROWS, NCOLS


def compute_ef(covs, drain_f, clay_grid, oc_grid):
    d_f  = drain_f
    t_f  = (1.0 + np.maximum(0, clay_grid - 20) * 0.015).astype("float32")
    sm   = covs["SM"]
    sm_f = np.where(sm > 0.35, 1.6,
           np.where(sm > 0.25, 1.2,
           np.where(sm > 0.15, 1.0, 0.7))).astype("float32")
    t    = covs["TEMP"]
    tp_f = np.where(t > 10, 1.3,
           np.where(t >  5, 1.1,
           np.where(t >  0, 1.0, 0.8))).astype("float32")
    ft_f = (1.0 + covs["FREEZE_THAW"] * 0.004).astype("float32")
    oc_f = (1.0 + np.maximum(0, oc_grid - 2.0) * 0.1).astype("float32")
    return np.clip(0.01 * d_f * t_f * sm_f * tp_f * ft_f * oc_f,
                   0.001, 0.06).astype("float32")


def run(covs, livestock_grids, ag_mask, drain_f, drain_poor,
        clay_grid, oc_grid, crop_ann):

    EF   = compute_ef(covs, drain_f, clay_grid, oc_grid)
    ndvi = covs["NDVI"]
    n_up = np.where(ndvi > 0.7, 0.85,
           np.where(ndvi > 0.5, 0.92,
           np.where(ndvi > 0.3, 1.00, 1.05))).astype("float32")

    N_app = np.zeros((NROWS, NCOLS), dtype="float32")
    N_pas = np.zeros((NROWS, NCOLS), dtype="float32")
    for animal, nex in N_EXCRETION.items():
        n_ex = (livestock_grids[animal] * nex).astype("float32")
        mf   = np.clip(MMS_FRAC[animal] * (0.5 + 0.5 * crop_ann), 0, 1).astype("float32")
        N_app += n_ex * mf
        N_pas += n_ex * (1 - mf)

    eff_N      = (N_app * n_up).astype("float32")
    N2O_direct = (eff_N * EF * (44/28) + N_pas * 0.02 * (44/28)).astype("float32")
    N2O_vol    = (eff_N * 0.20 * 0.01 * (44/28)).astype("float32")

    precip = covs["PRECIP"]
    f_leach = np.where(precip > 800, 0.30,
              np.where(precip > 600, 0.24,
              np.where(precip > 400, 0.15, 0.08))).astype("float32")
    drain_m = np.where(drain_poor, 1.5, 1.0).astype("float32")
    N2O_leach  = (eff_N * f_leach * 0.0075 * drain_m * (44/28)).astype("float32")
    N2O_indirect = (N2O_vol + N2O_leach).astype("float32")
    N2O_total    = (N2O_direct + N2O_indirect).astype("float32")
    N2O_ipcc     = (eff_N * 0.01 * (44/28) + N_pas * 0.02 * (44/28)
                    + N2O_vol + N2O_leach).astype("float32")

    for arr in [N2O_direct, N2O_indirect, N2O_total, N2O_ipcc, EF]:
        arr[~ag_mask] = np.nan

    # Fill NaN ag pixels with provincial mean
    from src.ag_mask import PROV_GRID
    n_nan = np.isnan(N2O_total[ag_mask]).sum()
    if n_nan / ag_mask.sum() > 0.10:
        for prov in np.unique(PROV_GRID[ag_mask]):
            pmask = (PROV_GRID == prov) & ag_mask
            m     = np.nanmean(N2O_total[pmask])
            if np.isfinite(m) and m > 0:
                fill = pmask & ~np.isfinite(N2O_total)
                N2O_total[fill]    = m
                N2O_direct[fill]   = np.nanmean(N2O_direct[pmask & np.isfinite(N2O_direct)])
                N2O_indirect[fill] = np.nanmean(N2O_indirect[pmask & np.isfinite(N2O_indirect)])
                N2O_ipcc[fill]     = np.nanmean(N2O_ipcc[pmask & np.isfinite(N2O_ipcc)])

    N2O_density      = (N2O_total / PIXEL_HA).astype("float32")
    EF_departure_pct = ((EF - 0.01) / 0.01 * 100).astype("float32")

    baseline = float(np.nansum(N2O_total)) / 1e6
    tier1    = float(np.nansum(N2O_ipcc))  / 1e6
    print(f"National N2O: {baseline:.3f} Gg  |  Tier 1: {tier1:.3f} Gg")

    return {
        "EF":               EF,
        "N2O_direct":       N2O_direct,
        "N2O_indirect":     N2O_indirect,
        "N2O_total":        N2O_total,
        "N2O_ipcc":         N2O_ipcc,
        "N2O_density":      N2O_density,
        "EF_departure_pct": EF_departure_pct,
        "national_Gg":      baseline,
        "tier1_Gg":         tier1,
    }
