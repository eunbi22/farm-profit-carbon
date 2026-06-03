"""
필지-격자 매핑 및 영농기 기상 feature 생성.

반환 DataFrame 인덱스: (uid, year)
컬럼: lat, lon, area_m2, cad_con_ra,
      ta_mean_m5~m9, rn_sum_m5~m9,
      ta_season_mean, rn_season_sum
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import KDTree
from config import (
    FARMMAP_CSV, FARMMAP_GEO, WEATHER_CSV,
    GROWING_MONTHS, TA_SCALE, RN_SCALE, DATA_START_DATE,
)


def load_farmmap_centroids() -> pd.DataFrame:
    """geojson에서 centroid 추출 + CSV에서 area/cad_con_ra 병합."""
    print("farmmap geojson 로딩 (centroid 계산)…")
    gdf = gpd.read_file(FARMMAP_GEO)
    gdf = gdf[gdf["is_farming"].isin([True, "Y", "true", 1])].copy()
    gdf["lon"] = gdf.geometry.centroid.x
    gdf["lat"] = gdf.geometry.centroid.y
    gdf["uid"] = gdf["uid"].astype(str)

    meta = pd.read_csv(
        FARMMAP_CSV, encoding="utf-8-sig",
        usecols=["uid", "area_m2", "cad_con_ra"],
        dtype={"uid": str},
    ).dropna(subset=["area_m2", "cad_con_ra"])
    meta["area_m2"]    = meta["area_m2"].astype(float)
    meta["cad_con_ra"] = meta["cad_con_ra"].astype(float)

    result = gdf[["uid", "lat", "lon"]].merge(meta, on="uid", how="inner")
    print(f"  경작 필지: {len(result):,}개")
    return result.reset_index(drop=True)


def load_weather() -> pd.DataFrame:
    """weather_grid 로딩, ta 스케일 적용, year/month 컬럼 추가."""
    print("기상 격자 데이터 로딩…")
    df = pd.read_csv(
        WEATHER_CSV,
        dtype={"date": str, "ny": int, "nx": int,
               "lat": float, "lon": float, "ta": float, "rn_day": float},
    )
    df["date"]   = pd.to_datetime(df["date"], format="%Y%m%d")
    df           = df[df["date"] >= DATA_START_DATE].copy()
    df["ta"]     = df["ta"]     * TA_SCALE
    df["rn_day"] = df["rn_day"] * RN_SCALE
    df["year"]   = df["date"].dt.year
    df["month"]  = df["date"].dt.month
    print(f"  {len(df):,}행, {df['date'].min().date()} ~ {df['date'].max().date()}")
    return df


def match_parcels_to_grid(centroids: pd.DataFrame,
                          weather: pd.DataFrame) -> pd.DataFrame:
    """KDTree로 각 필지 centroid에 가장 가까운 격자 (ny, nx) 부여."""
    grid_pts = weather[["ny", "nx", "lat", "lon"]].drop_duplicates(["ny", "nx"]).reset_index(drop=True)
    tree = KDTree(grid_pts[["lat", "lon"]].values)
    _, idx = tree.query(centroids[["lat", "lon"]].values)
    matched = grid_pts.iloc[idx][["ny", "nx"]].reset_index(drop=True)
    return pd.concat([centroids.reset_index(drop=True), matched], axis=1)


def aggregate_weather_by_grid_year(weather: pd.DataFrame) -> pd.DataFrame:
    """(ny, nx, year) 단위로 월별 + 영농기 전체 기상 집계."""
    gs = weather[weather["month"].isin(GROWING_MONTHS)].copy()

    monthly = (
        gs.groupby(["ny", "nx", "year", "month"])
          .agg(ta_mean=("ta", "mean"), rn_sum=("rn_day", "sum"))
          .reset_index()
    )
    monthly_wide = monthly.pivot_table(
        index=["ny", "nx", "year"],
        columns="month",
        values=["ta_mean", "rn_sum"],
    )
    monthly_wide.columns = [f"{v}_m{m}" for v, m in monthly_wide.columns]
    monthly_wide = monthly_wide.reset_index()

    seasonal = (
        gs.groupby(["ny", "nx", "year"])
          .agg(ta_season_mean=("ta", "mean"), rn_season_sum=("rn_day", "sum"))
          .reset_index()
    )
    return monthly_wide.merge(seasonal, on=["ny", "nx", "year"], how="outer")


def build_parcel_features() -> pd.DataFrame:
    """
    메인 엔트리포인트.
    (uid, year) 조합의 전체 feature DataFrame 반환.
    """
    centroids  = load_farmmap_centroids()
    weather    = load_weather()
    centroids  = match_parcels_to_grid(centroids, weather)
    grid_feats = aggregate_weather_by_grid_year(weather)

    years = sorted(weather["year"].unique())
    parcel_years = (
        centroids.assign(_k=1)
        .merge(pd.DataFrame({"year": years, "_k": 1}), on="_k")
        .drop(columns="_k")
    )
    result = parcel_years.merge(grid_feats, on=["ny", "nx", "year"], how="left")
    result = result.drop(columns=["ny", "nx"])
    print(f"Feature 테이블: {len(result):,}행 "
          f"({result['uid'].nunique():,}필지 × {len(years)}년)")
    return result
