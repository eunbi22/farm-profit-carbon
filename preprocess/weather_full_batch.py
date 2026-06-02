"""
weather_full_batch.py
─────────────────────────────────────────────────────────────────────────────
Batch-downloads KMA surface-grid NetCDF data for:
  • ta      (temperature)    – 4 obs/day (0000/0600/1200/1800) → daily mean
  • rn_day  (precipitation)  – 1 obs/day (0000 KST)

For each day, interpolates the Haenam grid points onto every farm-parcel
centroid (scipy linear griddata) so that each PNU gets its own value.

Output: data/haenam_weather_2020_2025.csv
Columns: pnu, date, element, value

Appends day-by-day to avoid memory pressure.
Auto-resumes: already-written (date, element) pairs are skipped.
─────────────────────────────────────────────────────────────────────────────
"""

import io
import os
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import xarray as xr
from scipy.interpolate import griddata

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
AUTH_KEY   = "5uVnBqgbS8KlZwaoG8vC8w"
DATE_START = "2020-05-01"
DATE_END   = "2025-12-31"

_DIR       = os.path.dirname(os.path.abspath(__file__))
GEOJSON    = os.path.join(_DIR, "data", "haenam_farmmap_20260527_1312.geojson")
OUTPUT_CSV = os.path.join(_DIR, "data", "haenam_weather_2020_2025.csv")

# Haenam bounding box (matches test.py)
LAT_MIN, LAT_MAX = 34.28, 34.67
LON_MIN, LON_MAX = 126.24, 126.75

RETRY_MAX   = 3    # retries per HTTP request
RETRY_DELAY = 5    # seconds between retries

# ─────────────────────────────────────────────────────────────────────────────
# KMA lat/lon grid map (fetched once, cached in memory)
# ─────────────────────────────────────────────────────────────────────────────
_LAT_LON_MAP: pd.DataFrame | None = None


def _parse_kma_latlon_text(text: str, value_col: str) -> pd.DataFrame:
    """Parse a KMA typ01 lat/lon text response into a ny×nx DataFrame."""
    first_nl     = text.index('\n')
    header_parts = text[:first_nl].replace('=', '').split(',')
    nx = int(header_parts[0].strip())
    ny = int(header_parts[1].strip())

    arr  = pd.read_csv(io.StringIO(text[first_nl + 1:]),
                       header=None, dtype=np.float64).values.flatten()
    vals = arr[~np.isnan(arr)]

    if len(vals) != nx * ny:
        raise ValueError(f"Grid size mismatch: expected {nx * ny}, got {len(vals)}")

    vals = vals.reshape(ny, nx)
    ny_idx, nx_idx = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    return pd.DataFrame({
        'ny': ny_idx.flatten().astype(np.int64),
        'nx': nx_idx.flatten().astype(np.int64),
        value_col: vals.flatten(),
    })


def get_latlon_map() -> pd.DataFrame:
    """Return the global KMA ny/nx → lat/lon lookup table, downloading once."""
    global _LAT_LON_MAP
    if _LAT_LON_MAP is not None:
        return _LAT_LON_MAP

    base = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-sfc_obs_latlon_api"
    print("[Init] Downloading KMA lat/lon grid map (one-time, ~10-20 s)...")

    lat_r = requests.get(f"{base}?latlon=lat&authKey={AUTH_KEY}", timeout=60)
    lon_r = requests.get(f"{base}?latlon=lon&authKey={AUTH_KEY}", timeout=60)

    if lat_r.status_code != 200:
        raise RuntimeError(f"Lat grid HTTP {lat_r.status_code}")
    if "인증 오류" in lat_r.text[:300]:
        raise RuntimeError(f"Auth error: {lat_r.text[:200]}")

    df_lat = _parse_kma_latlon_text(lat_r.text, 'lat')
    df_lon = _parse_kma_latlon_text(lon_r.text, 'lon')
    _LAT_LON_MAP = pd.merge(df_lat, df_lon, on=['ny', 'nx'])
    print(f"[Init] Grid map ready: {len(_LAT_LON_MAP):,} grid points\n")
    return _LAT_LON_MAP


# ─────────────────────────────────────────────────────────────────────────────
# Download one NetCDF → Haenam grid subset
# ─────────────────────────────────────────────────────────────────────────────
def fetch_haenam_grid(tm_str: str, element: str) -> pd.DataFrame | None:
    """
    Download one KMA NetCDF file, merge with lat/lon map, filter to Haenam bbox.

    Parameters
    ----------
    tm_str  : e.g. '202005010000'
    element : 'ta' or 'rn_day'

    Returns
    -------
    DataFrame(lat, lon, value) for Haenam grid points, or None on failure.
    """
    url     = (
        "https://apihub.kma.go.kr/api/typ01/url/sfc_grid_nc_down.php"
        f"?obs={element}&tm={tm_str}&authKey={AUTH_KEY}"
    )
    nc_path = f"_tmp_{element}_{tm_str}.nc"

    for attempt in range(1, RETRY_MAX + 1):
        try:
            r = requests.get(url, timeout=120)
            if r.status_code != 200:
                raise IOError(f"HTTP {r.status_code}")
            with open(nc_path, "wb") as f:
                f.write(r.content)
            break
        except Exception as exc:
            if attempt == RETRY_MAX:
                print(f"  [WARN] {tm_str}/{element}: download failed ({exc})")
                return None
            time.sleep(RETRY_DELAY)

    try:
        with xr.open_dataset(nc_path) as ds:
            data_var = list(ds.data_vars)[0]
            df = ds[data_var].to_dataframe().reset_index()

        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={data_var.lower(): 'value'})
        df = df.astype({'ny': 'int64', 'nx': 'int64'})

        grid_map = get_latlon_map()
        df = pd.merge(df, grid_map, on=['ny', 'nx'], how='inner')

        haenam = df[
            (df['lat'] >= LAT_MIN) & (df['lat'] <= LAT_MAX) &
            (df['lon'] >= LON_MIN) & (df['lon'] <= LON_MAX)
        ][['lat', 'lon', 'value']].dropna(subset=['value']).copy()

        return haenam if not haenam.empty else None

    except Exception as exc:
        print(f"  [WARN] {tm_str}/{element}: parse error ({exc})")
        return None

    finally:
        if os.path.exists(nc_path):
            os.remove(nc_path)


# ─────────────────────────────────────────────────────────────────────────────
# Spatial interpolation: Haenam grid → parcel centroids
# ─────────────────────────────────────────────────────────────────────────────
def interpolate_to_parcels(grid_df: pd.DataFrame,
                           parcel_pts: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate grid values onto parcel centroids.
    Falls back to nearest-neighbour for parcels outside the convex hull.

    Parameters
    ----------
    grid_df    : DataFrame with columns lat, lon, value
    parcel_pts : (N, 2) array of [lon, lat] centroid coordinates

    Returns
    -------
    (N,) array of interpolated values
    """
    src_pts  = grid_df[['lon', 'lat']].values
    src_vals = grid_df['value'].values

    result   = griddata(src_pts, src_vals, parcel_pts, method='linear')

    nan_mask = np.isnan(result)
    if nan_mask.any():
        result[nan_mask] = griddata(
            src_pts, src_vals, parcel_pts[nan_mask], method='nearest'
        )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_done_pairs(csv_path: str) -> set[tuple[str, str]]:
    """Return set of (date, element) pairs already written to the output CSV."""
    if not os.path.exists(csv_path):
        return set()
    try:
        existing = pd.read_csv(csv_path, usecols=['date', 'element'],
                               dtype=str, encoding='utf-8-sig')
        return set(zip(existing['date'], existing['element']))
    except Exception:
        return set()


def append_rows(rows: list[tuple], csv_path: str, write_header: bool) -> None:
    """Append a list of (pnu, date, element, value) tuples to the output CSV."""
    df = pd.DataFrame(rows, columns=['pnu', 'date', 'element', 'value'])
    df.to_csv(
        csv_path,
        index=False,
        mode='w' if write_header else 'a',
        header=write_header,
        encoding='utf-8-sig',
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── 1. Load parcel map once ───────────────────────────────────────────────
    print(f"[Init] Loading parcel GeoJSON ...")
    gdf = gpd.read_file(GEOJSON)
    gdf['pnu'] = gdf['pnu'].astype(str).str.strip().str.zfill(19)

    centroids  = gdf.geometry.centroid
    parcel_pts = np.column_stack([centroids.x.values, centroids.y.values])
    pnu_arr    = gdf['pnu'].values
    print(f"[Init] {len(gdf):,} parcels  |  CRS: {gdf.crs}")

    # ── 2. Pre-fetch lat/lon grid map ─────────────────────────────────────────
    get_latlon_map()

    # ── 3. Resume: find already-written (date, element) pairs ─────────────────
    done_pairs  = load_done_pairs(OUTPUT_CSV)
    need_header = not os.path.exists(OUTPUT_CSV)

    dates = pd.date_range(DATE_START, DATE_END, freq='D')
    print(f"[Batch] {len(dates)} days  ({DATE_START} -> {DATE_END})")
    if done_pairs:
        print(f"[Resume] {len(done_pairs)} (date, element) pairs already done, skipping.")
    print(f"[Output] {OUTPUT_CSV}\n")

    iterator = tqdm(dates, unit='day', desc='batch') if _HAS_TQDM else dates

    for date in iterator:
        date_str = date.strftime('%Y-%m-%d')
        date_ymd = date.strftime('%Y%m%d')

        if _HAS_TQDM:
            iterator.set_description(date_str)  # type: ignore[union-attr]
        else:
            print(f"  {date_str}", flush=True)

        day_rows: list[tuple] = []

        # ── rn_day: one download at 0000 KST ──────────────────────────────────
        if (date_str, 'rn_day') not in done_pairs:
            grid_rn = fetch_haenam_grid(f"{date_ymd}0000", 'rn_day')
            if grid_rn is not None and len(grid_rn) >= 3:
                vals = interpolate_to_parcels(grid_rn, parcel_pts)
                day_rows.extend(
                    (pnu, date_str, 'rn_day', round(float(v), 4))
                    for pnu, v in zip(pnu_arr, vals)
                )
            else:
                print(f"  [SKIP] {date_str} rn_day: insufficient grid data")

        # ── ta: 4 obs → daily mean ─────────────────────────────────────────────
        if (date_str, 'ta') not in done_pairs:
            ta_frames = []
            for hour in ('0000', '0600', '1200', '1800'):
                g = fetch_haenam_grid(f"{date_ymd}{hour}", 'ta')
                if g is not None:
                    ta_frames.append(g)

            if len(ta_frames) >= 1:
                ta_mean = (
                    pd.concat(ta_frames)
                    .groupby(['lat', 'lon'], as_index=False)['value']
                    .mean()
                )
                if len(ta_mean) >= 3:
                    vals = interpolate_to_parcels(ta_mean, parcel_pts)
                    day_rows.extend(
                        (pnu, date_str, 'ta', round(float(v), 4))
                        for pnu, v in zip(pnu_arr, vals)
                    )
                else:
                    print(f"  [SKIP] {date_str} ta: too few grid points after merge")
            else:
                print(f"  [SKIP] {date_str} ta: all hourly downloads failed")

        # ── Append this day's rows to CSV ──────────────────────────────────────
        if day_rows:
            append_rows(day_rows, OUTPUT_CSV, write_header=need_header)
            need_header = False

    print("\n[Done] Batch complete.")
    print(f"       Output: {OUTPUT_CSV}")

    if os.path.exists(OUTPUT_CSV):
        n_rows = sum(1 for _ in open(OUTPUT_CSV, encoding='utf-8-sig')) - 1
        print(f"       Rows written: {n_rows:,}")


if __name__ == '__main__':
    main()
