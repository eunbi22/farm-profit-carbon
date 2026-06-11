"""
Prophet 기상 기반 수확량 예측 (고정 파라미터, CV 없음).

실행: python train_prophet.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json, pickle, warnings
import numpy as np
import pandas as pd
from prophet import Prophet

from config import RESULT_DIR, TRAIN_YEARS_END
from dataset import prepare_county_series

warnings.filterwarnings("ignore")
SAVE_DIR  = os.path.join(RESULT_DIR, "models")
EXOG_COLS = ["ta_season_mean", "rn_season_sum"]
FIXED_PARAMS = {
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 1.0,
}


def _to_prophet_df(series_df: pd.DataFrame) -> pd.DataFrame:
    df = series_df.copy()
    df["ds"] = pd.to_datetime(df["year"].astype(str) + "-01-01")
    df["y"]  = df["yield_10a_kg"]
    return df[["ds", "y"] + EXOG_COLS].reset_index(drop=True)


def train_prophet(parcel_df, county_df):
    county_series = prepare_county_series(parcel_df, county_df)
    prophet_train = _to_prophet_df(county_series)

    m = Prophet(
        changepoint_prior_scale=FIXED_PARAMS["changepoint_prior_scale"],
        seasonality_prior_scale=FIXED_PARAMS["seasonality_prior_scale"],
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    for col in EXOG_COLS:
        m.add_regressor(col)
    m.fit(prophet_train)

    print(f"Prophet 파라미터: {FIXED_PARAMS}")

    with open(os.path.join(SAVE_DIR, "prophet.pkl"), "wb") as f:
        pickle.dump({
            "model":        m,
            "params":       FIXED_PARAMS,
            "train_series": county_series,
            "exog_cols":    EXOG_COLS,
        }, f)
    with open(os.path.join(RESULT_DIR, "loss_history_prophet.json"), "w") as f:
        json.dump({"trial_values": []}, f)

    print("Prophet 모델 저장 완료.")
    return m, county_series


if __name__ == "__main__":
    from features import build_parcel_features

    pf = build_parcel_features()
    pf["yield_per_10a"] = (
        pf["ta_season_mean"] * 5.0 + pf["rn_season_sum"] * 0.05 + 400
    ).clip(lower=100)
    county_df = (
        pf.groupby("year")
          .agg(area_ha=("area_m2", lambda x: x.sum() / 10000),
               yield_10a_kg=("yield_per_10a", "mean"))
          .reset_index()
    )
    train_prophet(pf, county_df)
