"""
Prophet 군 단위 생산량 예측 (Optuna 하이퍼파라미터 튜닝).
ta_season_mean, rn_season_sum 을 외부 regressor로 사용.
예측 결과는 disaggregate_yield로 필지에 분배.

실행: python train_prophet.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json, pickle, warnings
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from prophet import Prophet

from config import RESULT_DIR, OPTUNA_TRIALS, RANDOM_SEED, CV_FOLDS, TRAIN_YEARS_END
from dataset import prepare_county_series

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
SAVE_DIR  = os.path.join(RESULT_DIR, "models")
EXOG_COLS = ["ta_season_mean", "rn_season_sum"]


def _to_prophet_df(series_df: pd.DataFrame) -> pd.DataFrame:
    """Prophet 형식: ds(연도 → 연초), y, regressor 컬럼."""
    df = series_df.copy()
    df["ds"] = pd.to_datetime(df["year"].astype(str) + "-01-01")
    df["y"]  = df["yield_10a_kg"]
    return df[["ds", "y"] + EXOG_COLS].reset_index(drop=True)


def _fit_prophet(df_train: pd.DataFrame, params: dict) -> Prophet:
    m = Prophet(
        changepoint_prior_scale=params["changepoint_prior_scale"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    for col in EXOG_COLS:
        m.add_regressor(col)
    m.fit(df_train)
    return m


def _objective(trial, county_series, cv_splits):
    params = {
        "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale",
                                                        0.001, 1.0, log=True),
        "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale",
                                                        0.01, 10.0, log=True),
    }
    prophet_df = _to_prophet_df(county_series)
    years = sorted(county_series["year"].unique())

    fold_rmses = []
    for tr_years, val_years in cv_splits:
        tr_df  = prophet_df[prophet_df["ds"].dt.year.isin(tr_years)]
        val_df = prophet_df[prophet_df["ds"].dt.year.isin(val_years)]
        if len(tr_df) < 2:
            return float("inf")
        try:
            m    = _fit_prophet(tr_df, params)
            pred = m.predict(val_df[["ds"] + EXOG_COLS])["yhat"].values
            fold_rmses.append(np.sqrt(mean_squared_error(val_df["y"].values, pred)))
        except Exception:
            return float("inf")
    return float(np.mean(fold_rmses)) if fold_rmses else float("inf")


def train_prophet(parcel_df, county_df):
    county_series = prepare_county_series(parcel_df, county_df)
    years = sorted(county_series["year"].unique())

    tscv = TimeSeriesSplit(n_splits=CV_FOLDS)
    cv_splits = [
        ([years[i] for i in tr], [years[i] for i in val])
        for tr, val in tscv.split(years)
    ]

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(
        lambda t: _objective(t, county_series, cv_splits),
        n_trials=OPTUNA_TRIALS,
        show_progress_bar=True,
    )

    best_p = study.best_params
    print(f"\n최적 Prophet 파라미터: {best_p}")
    print(f"최적 CV RMSE: {study.best_value:.2f} kg/10a")

    train_series = county_series[county_series["year"] <= TRAIN_YEARS_END]
    prophet_train = _to_prophet_df(train_series)
    final = _fit_prophet(prophet_train, best_p)

    with open(os.path.join(SAVE_DIR, "prophet.pkl"), "wb") as f:
        pickle.dump({
            "model":        final,
            "params":       best_p,
            "train_series": train_series,
            "exog_cols":    EXOG_COLS,
        }, f)
    with open(os.path.join(RESULT_DIR, "loss_history_prophet.json"), "w") as f:
        json.dump({"trial_values": [t.value for t in study.trials]}, f)

    print("Prophet 모델 저장 완료.")
    return final, county_series


if __name__ == "__main__":
    from features import build_parcel_features
    from disaggregate import (
        load_county_production, fit_county_model,
        compute_parcel_weights, disaggregate_yield,
    )

    pf        = build_parcel_features()
    county_df = load_county_production()
    cw        = pf.groupby("year")[["ta_season_mean", "rn_season_sum"]].mean().reset_index()
    reg       = fit_county_model(county_df, cw)
    pf_w      = compute_parcel_weights(pf, reg["a"], reg["b"], reg["c"])
    parcel_df = disaggregate_yield(pf_w, county_df)
    train_prophet(parcel_df, county_df)
