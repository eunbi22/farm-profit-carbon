"""
SARIMAX 기상 기반 수확량 예측 (고정 파라미터, CV 없음).

실행: python train_sarimax.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json, pickle, warnings
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import RESULT_DIR, TRAIN_YEARS_END
from dataset import prepare_county_series

warnings.filterwarnings("ignore")
SAVE_DIR  = os.path.join(RESULT_DIR, "models")
EXOG_COLS = ["ta_season_mean", "rn_season_sum"]
FIXED_ORDER = (0, 0, 0)


def train_sarimax(parcel_df, county_df):
    county_series = prepare_county_series(parcel_df, county_df)

    endog = county_series["yield_10a_kg"].astype(float).values
    exog  = county_series[EXOG_COLS].astype(float).values
    final = SARIMAX(endog, exog=exog, order=FIXED_ORDER,
                    trend="c",
                    enforce_stationarity=False,
                    enforce_invertibility=False).fit(disp=False)

    print(f"SARIMAX order={FIXED_ORDER}  AIC={final.aic:.2f}")

    with open(os.path.join(SAVE_DIR, "sarimax.pkl"), "wb") as f:
        pickle.dump({
            "model":        final,
            "params":       {"p": 0, "d": 0, "q": 0},
            "train_series": county_series,
            "exog_cols":    EXOG_COLS,
        }, f)
    with open(os.path.join(RESULT_DIR, "loss_history_sarimax.json"), "w") as f:
        json.dump({"trial_values": [float(final.aic)]}, f)

    print("SARIMAX 모델 저장 완료.")
    return final, county_series


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
    train_sarimax(pf, county_df)
