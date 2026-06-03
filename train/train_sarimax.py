"""
SARIMAX 군 단위 생산량 예측 (Optuna p,d,q 탐색).
연간 데이터 6개 포인트이므로 단순 ARIMAX 수준으로 운용.
예측 결과는 disaggregate_yield로 필지에 분배.

실행: python train_sarimax.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json, pickle, warnings
import numpy as np
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import RESULT_DIR, OPTUNA_TRIALS, RANDOM_SEED, CV_FOLDS, TRAIN_YEARS_END
from dataset import prepare_county_series

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
SAVE_DIR = os.path.join(RESULT_DIR, "models")
EXOG_COLS = ["ta_season_mean", "rn_season_sum"]


def _fit(series_df, p, d, q):
    endog = series_df["yield_10a_kg"].astype(float).values
    exog  = series_df[EXOG_COLS].astype(float).values
    return SARIMAX(endog, exog=exog, order=(p, d, q),
                   trend="c",
                   enforce_stationarity=False,
                   enforce_invertibility=False).fit(disp=False)


def _objective(trial, county_series, cv_splits):
    p = trial.suggest_int("p", 0, 2)
    d = trial.suggest_int("d", 0, 1)
    q = trial.suggest_int("q", 0, 2)

    fold_rmses = []
    for tr_years, val_years in cv_splits:
        tr_df  = county_series[county_series["year"].isin(tr_years)]
        val_df = county_series[county_series["year"].isin(val_years)]
        if len(tr_df) < max(p, q) + 2:
            return float("inf")
        try:
            res  = _fit(tr_df, p, d, q)
            pred = res.forecast(steps=len(val_df),
                                exog=val_df[EXOG_COLS].astype(float).values)
            rmse = np.sqrt(mean_squared_error(val_df["yield_10a_kg"].values, pred))
            fold_rmses.append(rmse)
        except Exception:
            return float("inf")
    return float(np.mean(fold_rmses)) if fold_rmses else float("inf")


def train_sarimax(parcel_df, county_df):
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
    print(f"\n최적 SARIMAX order: p={best_p['p']}, d={best_p['d']}, q={best_p['q']}")
    print(f"최적 CV RMSE: {study.best_value:.2f} kg/10a")

    train_series = county_series[county_series["year"] <= TRAIN_YEARS_END]
    final = _fit(train_series, best_p["p"], best_p["d"], best_p["q"])

    with open(os.path.join(SAVE_DIR, "sarimax.pkl"), "wb") as f:
        pickle.dump({
            "model":        final,
            "params":       best_p,
            "train_series": train_series,
            "exog_cols":    EXOG_COLS,
        }, f)
    with open(os.path.join(RESULT_DIR, "loss_history_sarimax.json"), "w") as f:
        json.dump({"trial_values": [t.value for t in study.trials]}, f)

    print("SARIMAX 모델 저장 완료.")
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
    train_sarimax(parcel_df, county_df)
