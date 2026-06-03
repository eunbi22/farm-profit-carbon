"""
XGBoost 필지별 생산량 예측 (Optuna 하이퍼파라미터 튜닝).
실행: python train_xgboost.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error

from config import RESULT_DIR, OPTUNA_TRIALS, RANDOM_SEED
from dataset import (
    load_dataset, split_train_test, get_cv_splits,
    fit_scalers, prepare_tabular, ALL_FEATURES, TARGET,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
SAVE_DIR = os.path.join(RESULT_DIR, "models")


def _objective(trial, train_df, scaler_X, scaler_y, cv_splits):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 800),
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": RANDOM_SEED,
        "tree_method": "hist",
        "verbosity": 0,
    }
    fold_rmses = []
    for tr_mask, val_mask in cv_splits:
        X_tr,  y_tr  = prepare_tabular(train_df[tr_mask],  scaler_X, scaler_y)
        X_val, y_val = prepare_tabular(train_df[val_mask], scaler_X, scaler_y)
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  early_stopping_rounds=30,
                  verbose=False)
        pred = model.predict(X_val)
        fold_rmses.append(np.sqrt(mean_squared_error(y_val, pred)))
    return float(np.mean(fold_rmses))


def train_xgboost(parcel_df):
    df       = load_dataset(parcel_df)
    train_df, test_df = split_train_test(df)
    scaler_X, scaler_y = fit_scalers(train_df)
    cv_splits = get_cv_splits(train_df)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(
        lambda t: _objective(t, train_df, scaler_X, scaler_y, cv_splits),
        n_trials=OPTUNA_TRIALS,
        show_progress_bar=True,
    )

    best_p = study.best_params
    best_p.update({"random_state": RANDOM_SEED, "tree_method": "hist", "verbosity": 0})
    print(f"\n최적 XGBoost 파라미터: {best_p}")
    print(f"최적 CV RMSE (scaled): {study.best_value:.4f}")

    X_tr, y_tr = prepare_tabular(train_df, scaler_X, scaler_y)
    final = xgb.XGBRegressor(**best_p)
    final.fit(X_tr, y_tr)

    final.save_model(os.path.join(SAVE_DIR, "xgboost.json"))
    with open(os.path.join(RESULT_DIR, "metrics", "xgboost_best_params.json"), "w") as f:
        json.dump(best_p, f, indent=2, ensure_ascii=False)
    with open(os.path.join(RESULT_DIR, "loss_history_xgboost.json"), "w") as f:
        json.dump({"trial_values": [t.value for t in study.trials]}, f)

    print("XGBoost 모델 저장 완료.")
    return final, scaler_X, scaler_y, test_df


if __name__ == "__main__":
    from features import build_parcel_features
    from disaggregate import (
        load_county_production, fit_county_model,
        compute_parcel_weights, disaggregate_yield,
    )

    pf         = build_parcel_features()
    county_df  = load_county_production()
    cw         = pf.groupby("year")[["ta_season_mean", "rn_season_sum"]].mean().reset_index()
    reg        = fit_county_model(county_df, cw)
    pf_w       = compute_parcel_weights(pf, reg["a"], reg["b"], reg["c"])
    parcel_df  = disaggregate_yield(pf_w, county_df)
    train_xgboost(parcel_df)
