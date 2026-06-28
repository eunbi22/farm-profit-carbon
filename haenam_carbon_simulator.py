"""
HAENAM COUNTY IPCC TIER 1 METHANE REDUCTION SIMULATOR
======================================================
Uses FarmMap (팜맵) paddy parcel data to estimate CH4 reduction potential
from switching continuous flooding → intermittent irrigation, then integrates
XGBoost yield prediction to compare rice-only vs rice+carbon revenue.

IPCC references:
  - EF_base: IPCC 2019 Refinement, Vol.4 Ch.5, Table 5.1
  - SF_w:    IPCC 2019 Refinement, Vol.4 Ch.5, Table 5.2
  - SF_p:    IPCC 2019 Refinement, Vol.4 Ch.5, Table 5.3
  - SF_o:    IPCC 2019 Refinement, Vol.4 Ch.5, Table 5.5
  - GWP=27:  IPCC AR6 WGI Table 7.SM.7 (100-yr GWP, CH4 fossil-free)
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ── paths (always relative to this script file, regardless of cwd) ─────────────
_BASE      = Path(__file__).parent
INPUT_CSV  = _BASE / "data" / "haenam_farmmap_논.csv"
OUTPUT_CSV = _BASE / "data" / "haenam_carbon_simulation_results.csv"

# ── IPCC Tier 1 constants ──────────────────────────────────────────────────────
EF_BASE      = 1.6   # kg CH4 ha⁻¹ day⁻¹  (IPCC 2019, Table 5.1 default)
SF_W_BASE    = 1.0   # scaling factor – continuous flooding (pre-season flooded)
SF_W_IMPR    = 0.5   # scaling factor – intermittent irrigation (single aeration)
SF_P         = 1.0   # scaling factor – continuous paddy type (default)
SF_O         = 1.9   # scaling factor – straw incorporation (IPCC 2019, Table 5.5)
GROWING_DAYS = 130   # typical Korean single-crop growing season (days)

# ── economic constants ─────────────────────────────────────────────────────────
GWP_CH4          = 27      # AR6 100-yr GWP for biogenic CH4 (IPCC AR6)
CARBON_PRICE_KRW = 8_000   # KRW per tonne CO2eq (Korea ETS reference)
RICE_PRICE_KRW   = 2_100   # KRW per kg (KAMIS wholesale ~42,000 KRW / 20 kg)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/6] Loading FarmMap parcel data...")
t0 = time.perf_counter()

df = pd.read_csv(INPUT_CSV, usecols=["uid", "area_m2", "cad_con_ra"])
df = df.dropna(subset=["area_m2", "cad_con_ra"]).reset_index(drop=True)

print(f"      Loaded {len(df):,} parcels in {time.perf_counter()-t0:.2f}s")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — IPCC TIER 1 CH4 CALCULATION  (fully vectorized, zero loops)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/6] Computing IPCC Tier 1 CH4 emissions (vectorized)...")
t1 = time.perf_counter()

# Convert parcel area from m² → hectares
df["area_ha"] = df["area_m2"] / 10_000.0

# CH4 formula: EF_base × SF_w × SF_p × SF_o × growing_days × area_ha
# SF_p and SF_o are constants applied to every parcel simultaneously via numpy
common_factor = EF_BASE * SF_P * SF_O * GROWING_DAYS  # scalar, computed once

df["CH4_baseline"] = common_factor * SF_W_BASE * df["area_ha"]   # continuous flooding
df["CH4_improved"] = common_factor * SF_W_IMPR * df["area_ha"]   # intermittent irrigation

# ── carbon reduction metrics ──────────────────────────────────────────────────
df["delta_CH4"]     = df["CH4_baseline"] - df["CH4_improved"]            # kg CH4 saved
df["delta_CO2_ton"] = df["delta_CH4"] * GWP_CH4 / 1_000.0               # tonnes CO2eq
df["carbon_revenue_krw"] = df["delta_CO2_ton"] * CARBON_PRICE_KRW       # KRW

print(f"      Done in {time.perf_counter()-t1:.4f}s")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — SYNTHETIC YIELD LABELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/6] Generating synthetic yield labels (Korean paddy distribution)...")
rng = np.random.default_rng(42)

# Simulates realistic Korean paddy yield ~480–550 kg/10a
# Larger area → slightly higher yield (economies of field size)
# Higher cadastral conformity → higher yield (better-managed parcel)
noise = rng.normal(0, 20, size=len(df))
df["yield_per_10a"] = (
    480
    + (df["area_ha"] * 12)
    + (df["cad_con_ra"] * 0.8)
    + noise
)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — XGBOOST YIELD PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/6] Training XGBoost yield prediction model...")
t2 = time.perf_counter()

FEATURES = ["area_ha", "cad_con_ra", "CH4_baseline", "delta_CO2_ton"]
X = df[FEATURES].values
y = df["yield_per_10a"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = XGBRegressor(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
r2   = r2_score(y_test, y_pred_test)

# Predict yield for ALL 69,810 parcels (vectorized XGBoost inference)
df["yield_per_10a"] = model.predict(X)

print(f"      Training + inference done in {time.perf_counter()-t2:.2f}s")
print(f"      RMSE: {rmse:.2f} kg/10a  |  R²: {r2:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — REVENUE COMPARISON ENGINE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/6] Computing revenue comparison (Option A vs B)...")
t3 = time.perf_counter()

# Option A — sell rice only
# yield_per_10a [kg/10a] × area [ha] × 10 [10a/ha] = total yield [kg]
df["revenue_A"] = (df["yield_per_10a"] * df["area_ha"] * 10) * RICE_PRICE_KRW

# Option B — sell rice + carbon credit (purely additive)
df["revenue_B"] = df["revenue_A"] + df["carbon_revenue_krw"]

# B is always better because carbon revenue is strictly positive;
# column kept for structural completeness and future threshold modelling
df["better_option"] = np.where(df["revenue_B"] > df["revenue_A"], "B", "A")

print(f"      Done in {time.perf_counter()-t3:.4f}s")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6/6] Saving per-parcel results...")

output_cols = [
    "uid",          # parcel identifier (→ renamed parcel_id in output)
    "area_ha",
    "cad_con_ra",
    "CH4_baseline",
    "CH4_improved",
    "delta_CH4",
    "delta_CO2_ton",
    "carbon_revenue_krw",
    "yield_per_10a",
    "revenue_A",
    "revenue_B",
    "better_option",
]

df[output_cols].rename(columns={"uid": "parcel_id"}).to_csv(OUTPUT_CSV, index=False)
print(f"      Saved → {OUTPUT_CSV}")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
total_parcels      = len(df)
total_area_ha      = df["area_ha"].sum()

total_CH4_reduced  = df["delta_CH4"].sum() / 1_000          # tonnes CH4
total_CO2eq        = df["delta_CO2_ton"].sum()               # tonnes CO2eq
avg_CO2eq          = df["delta_CO2_ton"].mean()

total_carbon_rev   = df["carbon_revenue_krw"].sum()          # KRW
avg_carbon_rev     = df["carbon_revenue_krw"].mean()
avg_rev_A          = df["revenue_A"].mean()
avg_rev_B          = df["revenue_B"].mean()
avg_uplift         = avg_rev_B - avg_rev_A
pct_uplift         = avg_uplift / avg_rev_A * 100

print("""
================================================
       HAENAM COUNTY CARBON SIMULATOR
       IPCC Tier 1 | GWP=27 | SF_w: 1.0→0.5
================================================""")
print(f"Total parcels analyzed          : {total_parcels:,}")
print(f"Total area analyzed             : {total_area_ha:,.1f} ha")
print("""
--- CARBON REDUCTION POTENTIAL ---""")
print(f"Total CH4 reduced               : {total_CH4_reduced:,.1f} tonnes")
print(f"Total CO2eq reduced             : {total_CO2eq:,.1f} tonnes CO2eq")
print(f"Avg CO2eq reduced per parcel    : {avg_CO2eq:.2f} tonnes")
print("""
--- FINANCIAL VALUE CREATED ---""")
print(f"Carbon credit revenue (total)   : {total_carbon_rev/1e9:.1f} billion KRW")
print(f"Avg carbon revenue per parcel   : {avg_carbon_rev:,.0f} KRW")
print(f"Avg rice revenue per parcel (A) : {avg_rev_A:,.0f} KRW")
print(f"Avg total revenue per parcel (B): {avg_rev_B:,.0f} KRW")
print(f"Avg revenue uplift (B-A)        : {avg_uplift:,.0f} KRW (+{pct_uplift:.1f}%)")
print("""
--- AI YIELD MODEL PERFORMANCE ---""")
print(f"XGBoost RMSE                    : {rmse:.2f} kg/10a")
print(f"XGBoost R²                      : {r2:.3f}")
print("================================================\n")
