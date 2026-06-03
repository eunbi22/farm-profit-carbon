"""
시군 단위 생산량 파싱 + 선형회귀로 a, b, c 추정.
군 총생산량을 필지별 가중치로 분배.

분배 공식:
    W_i = area_m2 × (cad_con_ra/100) × max(a*ta + b*rn + c, ε)
    Y_i = 군_총생산량_kg × W_i / Σ W_i
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle, os
from config import PRODUCTION_CSV, RESULT_DIR, DATA_START_DATE


def load_county_production() -> pd.DataFrame:
    """
    2행 헤더의 wide-format CSV를 파싱 →
    columns: year, area_ha, yield_10a_kg, total_ton
    """
    raw = pd.read_csv(PRODUCTION_CSV, header=None, encoding="utf-8-sig")
    year_row = raw.iloc[0].tolist()
    data_row = raw.iloc[2].tolist()   # 해남군 행

    records, col = [], 2
    while col + 2 < len(year_row):
        try:
            year = int(str(year_row[col]).strip())
        except (ValueError, TypeError):
            col += 3
            continue
        records.append({
            "year":        year,
            "area_ha":     float(data_row[col]),
            "yield_10a_kg": float(data_row[col + 1]),
            "total_ton":   float(data_row[col + 2]),
        })
        col += 3

    df = pd.DataFrame(records).dropna()
    start_year = pd.to_datetime(DATA_START_DATE).year
    df = df[df["year"] >= start_year].reset_index(drop=True)
    print(f"군 생산량 데이터: {df['year'].min()}~{df['year'].max()}년 ({len(df)}개 연도)")
    return df


def fit_county_model(county_df: pd.DataFrame,
                     weather_county: pd.DataFrame) -> dict:
    """
    회귀: yield_10a_kg = a*ta_season_mean + b*rn_season_sum + c
    weather_county: year, ta_season_mean, rn_season_sum (필지 평균 → 군 대표값)
    """
    merged = county_df.merge(weather_county, on="year", how="inner")
    if len(merged) < 3:
        raise ValueError(f"회귀에 필요한 샘플 부족: {len(merged)}개")

    X = merged[["ta_season_mean", "rn_season_sum"]].values
    y = merged["yield_10a_kg"].values
    reg = LinearRegression().fit(X, y)

    result = {
        "a": float(reg.coef_[0]),
        "b": float(reg.coef_[1]),
        "c": float(reg.intercept_),
        "model": reg,
        "r2": float(reg.score(X, y)),
        "years_used": merged["year"].tolist(),
    }
    print(f"군 회귀 결과: a={result['a']:.4f}, b={result['b']:.6f}, "
          f"c={result['c']:.2f}, R²={result['r2']:.3f}  (n={len(merged)})")

    save_path = os.path.join(RESULT_DIR, "models", "county_regression.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(result, f)
    return result


def compute_parcel_weights(parcel_df: pd.DataFrame,
                           a: float, b: float, c: float,
                           eps: float = 1e-6) -> pd.DataFrame:
    """
    W_i = area_m2 × (cad_con_ra/100) × max(a*ta + b*rn + c, eps)
    weather_score 및 weight 컬럼 추가하여 반환.
    """
    f = (a * parcel_df["ta_season_mean"] + b * parcel_df["rn_season_sum"] + c).clip(lower=eps)
    df = parcel_df.copy()
    df["weather_score"] = f
    df["weight"]        = df["area_m2"] * (df["cad_con_ra"] / 100.0) * f
    return df


def disaggregate_yield(parcel_df: pd.DataFrame,
                       county_df: pd.DataFrame) -> pd.DataFrame:
    """
    연도별로 군 총생산량을 필지 weight 비율에 따라 분배.
    추가 컬럼: yield_kg, yield_per_10a
    """
    county_kg = (county_df.set_index("year")["total_ton"] * 1000).to_dict()

    frames = []
    for year, grp in parcel_df.groupby("year"):
        if year not in county_kg:
            continue
        w_sum = grp["weight"].sum()
        grp = grp.copy()
        grp["yield_kg"]      = county_kg[year] * grp["weight"] / w_sum
        grp["yield_per_10a"] = grp["yield_kg"] / (grp["area_m2"] / 1000.0)
        frames.append(grp)

    result = pd.concat(frames, ignore_index=True)
    print(f"분배 완료: {result['year'].nunique()}개 연도, {len(result):,}개 (필지×연도) 레코드")
    return result
