"""
Haenam Farm Parcel Weather Interpolation
-----------------------------------------
1. Merge farm parcel CSV attributes into the GeoJSON.
2. Build a synthetic spatial weather grid seeded from the real
   Haenam station (261) values for the most recent available date.
3. Interpolate (scipy linear griddata) to each parcel centroid.
4. Visualize with Korean labels and save results.
"""

import os
import warnings

import geopandas as gpd
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR        = os.path.join(os.path.dirname(__file__), "data")
FARMMAP_CSV     = os.path.join(DATA_DIR, "haenam_farmmap_20260527_1312.csv")
FARMMAP_GEOJSON = os.path.join(DATA_DIR, "haenam_farmmap_20260527_1312.geojson")
TEMP_CSV        = os.path.join(DATA_DIR, "ta_20260601172048.csv")
PRECIP_CSV      = os.path.join(DATA_DIR, "rn_20260601171702.csv")
OUTPUT_GEOJSON  = os.path.join(DATA_DIR, "haenam_farmmap_weather_interpolated.geojson")
OUTPUT_PNG      = os.path.join(DATA_DIR, "haenam_weather_interpolation.png")


# ──────────────────────────────────────────────────────────────────────────────
# Korean font setup (Windows: Malgun Gothic; Mac/Linux: NanumGothic fallback)
# ──────────────────────────────────────────────────────────────────────────────
def _setup_korean_font():
    candidates = ["NanumGothic", "Malgun Gothic", "AppleGothic", "UnDotum",
                  "Nanum Gothic", "Gulim", "Batang"]
    available  = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            print(f"Korean font: {name}")
            return
    # Last resort: any font whose name contains a Korean-font keyword
    for f in fm.fontManager.ttflist:
        if any(k in f.name for k in ("Gothic", "Nanum", "Batang", "Gulim", "Gungsuh")):
            plt.rcParams["font.family"] = f.name
            print(f"Korean font (fallback): {f.name}")
            return
    print("Warning: No Korean font detected — labels may render as boxes.")

_setup_korean_font()
plt.rcParams["axes.unicode_minus"] = False


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 · Load and merge farm parcel data
# ──────────────────────────────────────────────────────────────────────────────
print("\n[Step 1] Loading farm parcel data …")

# CSV — cp949; pnu must stay as string to preserve leading zeros
csv_df = pd.read_csv(FARMMAP_CSV, dtype={"pnu": str}, encoding="cp949")
csv_df["pnu"] = csv_df["pnu"].str.strip().str.zfill(19)

# GeoJSON — already UTF-8; geopandas handles encoding
gdf = gpd.read_file(FARMMAP_GEOJSON)
gdf["pnu"] = gdf["pnu"].astype(str).str.strip().str.zfill(19)

# Merge only columns not already present in the GeoJSON (pnu is the join key)
gdf_cols  = set(gdf.columns)          # includes 'pnu', 'geometry', etc.
new_cols  = [c for c in csv_df.columns if c not in gdf_cols]
gdf = gdf.merge(csv_df[["pnu"] + new_cols], on="pnu", how="left")

print(f"  Features     : {len(gdf):,}")
print(f"  CRS          : {gdf.crs}")
print(f"  Bounds (lon) : {gdf.total_bounds[0]:.4f} → {gdf.total_bounds[2]:.4f}")
print(f"  Bounds (lat) : {gdf.total_bounds[1]:.4f} → {gdf.total_bounds[3]:.4f}")
print(f"  Merged cols  : {new_cols}")


# ──────────────────────────────────────────────────────────────────────────────
# Step 2a · Parse real weather station data (station 261 = Haenam)
# ──────────────────────────────────────────────────────────────────────────────
print("\n[Step 2a] Parsing weather station CSVs …")

# Temperature — rows 0-6 are metadata; date column has a leading tab
temp_df = pd.read_csv(
    TEMP_CSV, encoding="cp949", skiprows=7,
    header=0, names=["date", "station", "ta_avg", "ta_min", "ta_max"],
)
temp_df["date"]   = temp_df["date"].astype(str).str.strip()
temp_df["ta_avg"] = pd.to_numeric(temp_df["ta_avg"], errors="coerce")

# Precipitation
precip_df = pd.read_csv(
    PRECIP_CSV, encoding="cp949", skiprows=7,
    header=0, names=["date", "station", "rn"],
)
precip_df["date"] = precip_df["date"].astype(str).str.strip()
precip_df["rn"]   = pd.to_numeric(precip_df["rn"], errors="coerce").fillna(0.0)

# Most recent date with a valid temperature reading
latest = temp_df.dropna(subset=["ta_avg"]).iloc[-1]
ref_date   = latest["date"]
base_temp  = float(latest["ta_avg"])

precip_row = precip_df[precip_df["date"] == ref_date]["rn"]
base_precip = float(precip_row.iloc[0]) if not precip_row.empty else 0.0

print(f"  Reference date      : {ref_date}")
print(f"  Station temperature : {base_temp} °C")
print(f"  Station precip.     : {base_precip} mm")


# ──────────────────────────────────────────────────────────────────────────────
# Step 2b · Build synthetic spatial weather grid
#
# The source data contains one station.  A spatial interpolation requires
# multiple grid points, so we construct a 6×6 placeholder grid that:
#   • spans the full farm-parcel extent with a margin so edge parcels
#     fall inside the interpolation convex hull (no NaN from griddata);
#   • seeds every grid-point temperature / precipitation from the real
#     station value and adds geographically plausible gradients + noise.
# ──────────────────────────────────────────────────────────────────────────────
print("\n[Step 2b] Building synthetic weather grid …")

minx, miny, maxx, maxy = gdf.total_bounds   # EPSG:4326 lon/lat
MARGIN = 0.10                                # degrees; ensures full coverage

lons = np.linspace(minx - MARGIN, maxx + MARGIN, 6)
lats = np.linspace(miny - MARGIN, maxy + MARGIN, 6)
lon_grid, lat_grid = np.meshgrid(lons, lats)

np.random.seed(42)

# Temperature: gentle inland warming (north/east) + small random noise
temp_field = (
    base_temp
    + (lat_grid - miny) * 0.8        # +0.8 °C per degree latitude northward
    + (lon_grid - minx) * 0.4        # +0.4 °C per degree longitude eastward
    + np.random.normal(0, 0.25, lon_grid.shape)
)

# Precipitation: coastal (west) higher, drops eastward + noise
precip_field = np.maximum(
    0.0,
    base_precip
    - (lon_grid - minx) * 3.0        # coastal gradient
    + np.random.normal(0, 1.0, lon_grid.shape)
)

grid_pts     = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
temp_vals    = temp_field.ravel()
precip_vals  = precip_field.ravel()

print(f"  Grid shape     : {lon_grid.shape[0]} × {lon_grid.shape[1]} "
      f"({grid_pts.shape[0]} points)")
print(f"  Temp range     : {temp_vals.min():.2f} -{temp_vals.max():.2f} °C")
print(f"  Precip range   : {precip_vals.min():.2f} -{precip_vals.max():.2f} mm")


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 · Interpolate grid → parcel centroids
# ──────────────────────────────────────────────────────────────────────────────
print("\n[Step 3] Interpolating weather to parcel centroids …")

centroids     = gdf.geometry.centroid
parcel_pts    = np.column_stack([centroids.x.values, centroids.y.values])

interp_temp   = griddata(grid_pts, temp_vals,   parcel_pts, method="linear")
interp_precip = griddata(grid_pts, precip_vals, parcel_pts, method="linear")

# Nearest-neighbour fill for any parcels outside the convex hull
nan_mask = np.isnan(interp_temp)
if nan_mask.any():
    interp_temp[nan_mask]   = griddata(grid_pts, temp_vals,
                                        parcel_pts[nan_mask], method="nearest")
    interp_precip[nan_mask] = griddata(grid_pts, precip_vals,
                                        parcel_pts[nan_mask], method="nearest")

gdf["interp_temp"]   = interp_temp
gdf["interp_precip"] = interp_precip

print(f"  Interpolated temp   : {gdf['interp_temp'].min():.2f} -"
      f"{gdf['interp_temp'].max():.2f} °C")
print(f"  Interpolated precip : {gdf['interp_precip'].min():.2f} -"
      f"{gdf['interp_precip'].max():.2f} mm")


# ──────────────────────────────────────────────────────────────────────────────
# Step 4 · Visualize
# ──────────────────────────────────────────────────────────────────────────────
print("\n[Step 4] Generating visualization …")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
fig.suptitle(
    f"해남군 농경지 필지별 기상 보간 결과  ({ref_date})",
    fontsize=15, fontweight="bold",
)

_scatter_kw = dict(s=55, marker="^", edgecolors="black",
                   linewidths=0.6, zorder=5, alpha=0.9)
_legend_kw  = dict(orientation="vertical", shrink=0.75, pad=0.02)

# — Temperature —
gdf.plot(
    column="interp_temp", ax=ax1, cmap="RdYlBu_r",
    legend=True,
    legend_kwds={"label": "평균기온 (°C)", **_legend_kw},
    missing_kwds={"color": "lightgrey"},
    linewidth=0.0, edgecolor="none",
)
ax1.scatter(
    lon_grid.ravel(), lat_grid.ravel(),
    c=temp_vals, cmap="RdYlBu_r",
    vmin=gdf["interp_temp"].min(), vmax=gdf["interp_temp"].max(),
    label="격자점 (기상)", **_scatter_kw,
)
ax1.set_title("평균기온 보간", fontsize=13)
ax1.set_xlabel("경도 (°E)")
ax1.set_ylabel("위도 (°N)")
ax1.legend(loc="lower right", fontsize=9)

# — Precipitation —
gdf.plot(
    column="interp_precip", ax=ax2, cmap="Blues",
    legend=True,
    legend_kwds={"label": "강수량 (mm)", **_legend_kw},
    missing_kwds={"color": "lightgrey"},
    linewidth=0.0, edgecolor="none",
)
ax2.scatter(
    lon_grid.ravel(), lat_grid.ravel(),
    c=precip_vals, cmap="Blues",
    vmin=gdf["interp_precip"].min(), vmax=gdf["interp_precip"].max(),
    label="격자점 (기상)", **_scatter_kw,
)
ax2.set_title("강수량 보간", fontsize=13)
ax2.set_xlabel("경도 (°E)")
ax2.set_ylabel("위도 (°N)")
ax2.legend(loc="lower right", fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"  Saved map  : {OUTPUT_PNG}")
plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Step 5 · Save output GeoJSON
# ──────────────────────────────────────────────────────────────────────────────
print("\n[Step 5] Saving output GeoJSON …")
gdf.to_file(OUTPUT_GEOJSON, driver="GeoJSON")
print(f"  Saved data : {OUTPUT_GEOJSON}")
print(f"  Features   : {len(gdf):,}")
print("\nDone.")
