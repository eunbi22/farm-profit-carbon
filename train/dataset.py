"""
데이터 로딩, 분할, 모델별 포맷 변환.

분할 전략:
  - 전체: 연도 기준 8:2 → train(2020-2024) / test(2025)
  - train 내부: TimeSeriesSplit(3-fold) 로 CV
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import pickle, os
from config import (
    RESULT_DIR, TRAIN_YEARS_END, TEST_YEARS_START,
    CV_FOLDS, GROWING_MONTHS,
)

MONTHLY_TA   = [f"ta_mean_m{m}" for m in GROWING_MONTHS]
MONTHLY_RN   = [f"rn_sum_m{m}"  for m in GROWING_MONTHS]
SEASONAL     = ["ta_season_mean", "rn_season_sum"]
STATIC       = ["area_m2", "cad_con_ra", "lat", "lon"]
ALL_FEATURES = STATIC + SEASONAL + MONTHLY_TA + MONTHLY_RN
TARGET       = "yield_per_10a"
COUNTY_FEATS = SEASONAL   # SARIMAX / Prophet 입력


def load_dataset(parcel_df: pd.DataFrame) -> pd.DataFrame:
    required = ALL_FEATURES + [TARGET, "year", "uid"]
    df = parcel_df.dropna(subset=required).copy()
    print(f"학습 데이터: {len(df):,}행 (NaN 제거 후)")
    return df


def split_train_test(df: pd.DataFrame):
    train = df[df["year"] <= TRAIN_YEARS_END].copy()
    test  = df[df["year"] >= TEST_YEARS_START].copy()
    print(f"Train {train['year'].min()}~{train['year'].max()} ({len(train):,}행)  "
          f"| Test {test['year'].min()}~{test['year'].max()} ({len(test):,}행)")
    return train, test


def get_cv_splits(train_df: pd.DataFrame):
    """연도 기준 TimeSeriesSplit → (train_bool_mask, val_bool_mask) 리스트."""
    years = sorted(train_df["year"].unique())
    tscv  = TimeSeriesSplit(n_splits=CV_FOLDS)
    splits = []
    for tr_idx, val_idx in tscv.split(years):
        tr_years  = {years[i] for i in tr_idx}
        val_years = {years[i] for i in val_idx}
        splits.append((
            train_df["year"].isin(tr_years),
            train_df["year"].isin(val_years),
        ))
    return splits


def fit_scalers(train_df: pd.DataFrame):
    scaler_X = StandardScaler().fit(train_df[ALL_FEATURES])
    scaler_y = StandardScaler().fit(train_df[[TARGET]])
    for name, obj in [("scaler_X.pkl", scaler_X), ("scaler_y.pkl", scaler_y)]:
        with open(os.path.join(RESULT_DIR, "models", name), "wb") as f:
            pickle.dump(obj, f)
    return scaler_X, scaler_y


def load_scalers():
    out = {}
    for name in ("scaler_X.pkl", "scaler_y.pkl"):
        with open(os.path.join(RESULT_DIR, "models", name), "rb") as f:
            out[name.replace(".pkl", "")] = pickle.load(f)
    return out["scaler_X"], out["scaler_y"]


def prepare_tabular(df: pd.DataFrame, scaler_X, scaler_y=None):
    X = scaler_X.transform(df[ALL_FEATURES])
    y = scaler_y.transform(df[[TARGET]]).ravel() if scaler_y is not None else None
    return X, y


def prepare_lstm_sequences(df: pd.DataFrame, scaler_X, scaler_y=None):
    """
    반환:
      seq_input    (N, 5, 2)  – 월별 [avg_ta, sum_rn] May-Sep
      static_input (N, 4)     – [area, cad_con_ra, lat, lon] 정규화
      y            (N,)       – 타겟 (정규화)
    """
    X_all  = scaler_X.transform(df[ALL_FEATURES])
    n_stat = len(STATIC)
    static_input = X_all[:, :n_stat]

    ta_idx = [ALL_FEATURES.index(c) for c in MONTHLY_TA]
    rn_idx = [ALL_FEATURES.index(c) for c in MONTHLY_RN]
    seq_ta = X_all[:, ta_idx]   # (N, 5)
    seq_rn = X_all[:, rn_idx]   # (N, 5)
    seq_input = np.stack([seq_ta, seq_rn], axis=2)  # (N, 5, 2)

    y = scaler_y.transform(df[[TARGET]]).ravel() if scaler_y is not None else None
    return seq_input, static_input, y


def prepare_county_series(parcel_df: pd.DataFrame,
                          county_df: pd.DataFrame) -> pd.DataFrame:
    """SARIMAX/Prophet용: 연도별 군 평균 기상 + 군 10a당 생산량."""
    county_weather = (
        parcel_df.groupby("year")[SEASONAL].mean().reset_index()
    )
    return county_weather.merge(
        county_df[["year", "yield_10a_kg"]], on="year", how="inner"
    )
