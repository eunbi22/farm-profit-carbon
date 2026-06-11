"""
전체 모델 테스트셋 평가 + SHAP 분석.
각 모델 예측값, 메트릭, SHAP 값을 train/result/ 에 저장.

실행: python evaluate.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json, pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import xgboost as xgb
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_error

from config import RESULT_DIR, TEST_YEARS_START, RANDOM_SEED
from dataset import (
    load_dataset, split_train_test, load_scalers,
    prepare_tabular, prepare_lstm_sequences, prepare_county_series,
    ALL_FEATURES, TARGET,
)
from train_lstm import build_model

SAVE_DIR = os.path.join(RESULT_DIR, "models")


# ─── 유틸 ────────────────────────────────────────────────────────────────────

def _metrics(y_true, y_pred, label):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    print(f"[{label}]  RMSE={rmse:.2f}  MAE={mae:.2f}  MAPE={mape:.2f}%")
    return {"model": label, "RMSE": rmse, "MAE": mae, "MAPE": mape}


def _inverse_y(arr, scaler_y):
    return scaler_y.inverse_transform(arr.reshape(-1, 1)).ravel()


# ─── 각 모델 예측 ────────────────────────────────────────────────────────────

def predict_xgboost(test_df, scaler_X, scaler_y):
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(SAVE_DIR, "xgboost.json"))
    X_test, y_test = prepare_tabular(test_df, scaler_X, scaler_y)
    pred_s = model.predict(X_test)
    return _inverse_y(pred_s, scaler_y), _inverse_y(y_test, scaler_y), model


def predict_lstm(test_df, scaler_X, scaler_y):
    model = keras.models.load_model(os.path.join(SAVE_DIR, "lstm.keras"))
    seq, static, y_s = prepare_lstm_sequences(test_df, scaler_X, scaler_y)
    pred_s = model.predict([seq, static], verbose=0).ravel()
    return _inverse_y(pred_s, scaler_y), _inverse_y(y_s, scaler_y), model


def predict_sarimax(test_df, county_df):
    with open(os.path.join(SAVE_DIR, "sarimax.pkl"), "rb") as f:
        saved = pickle.load(f)
    county_series = prepare_county_series(test_df, county_df)
    test_county   = county_series[county_series["year"] >= TEST_YEARS_START]
    if len(test_county) == 0:
        true_parcel = test_df.sort_values("year")[TARGET].values
        return np.full(len(true_parcel), np.nan), true_parcel
    exog = test_county[saved["exog_cols"]].astype(float).values
    pred_10a_county = saved["model"].forecast(steps=len(test_county), exog=exog)

    # 군 예측값을 필지 면적 비례로 분배
    year_pred = dict(zip(test_county["year"].values, pred_10a_county))
    pred_list = []
    for year, grp in test_df.sort_values("year").groupby("year"):
        val = year_pred.get(year, np.nan)
        pred_list.extend([val] * len(grp))
    true_parcel = test_df.sort_values("year")[TARGET].values
    return np.array(pred_list), true_parcel


def predict_prophet(test_df, county_df):
    with open(os.path.join(SAVE_DIR, "prophet.pkl"), "rb") as f:
        saved = pickle.load(f)
    county_series = prepare_county_series(test_df, county_df)
    test_county   = county_series[county_series["year"] >= TEST_YEARS_START].copy()
    if len(test_county) == 0:
        true_parcel = test_df.sort_values("year")[TARGET].values
        return np.full(len(true_parcel), np.nan), true_parcel
    test_county["ds"] = pd.to_datetime(test_county["year"].astype(str) + "-01-01")
    pred_df_prophet = saved["model"].predict(test_county[["ds"] + saved["exog_cols"]])
    year_pred = dict(zip(test_county["year"].values, pred_df_prophet["yhat"].values))
    pred_list = []
    for year, grp in test_df.sort_values("year").groupby("year"):
        val = year_pred.get(year, np.nan)
        pred_list.extend([val] * len(grp))
    true_parcel = test_df.sort_values("year")[TARGET].values
    return np.array(pred_list), true_parcel


# ─── SHAP ────────────────────────────────────────────────────────────────────

def run_shap_xgboost(model, test_df, scaler_X):
    X_test, _ = prepare_tabular(test_df, scaler_X)
    explainer  = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_test)
    df = pd.DataFrame(sv, columns=ALL_FEATURES)
    df.to_csv(os.path.join(RESULT_DIR, "shap_xgboost.csv"), index=False)
    print("XGBoost SHAP 저장 완료.")
    return sv


def run_shap_lstm(model, test_df, scaler_X):
    seq, static, _ = prepare_lstm_sequences(test_df, scaler_X)
    bg_size = min(100, len(seq))
    np.random.seed(RANDOM_SEED)
    idx = np.random.choice(len(seq), bg_size, replace=False)

    bg_seq    = seq[idx].astype(np.float32)
    bg_static = static[idx].astype(np.float32)

    explainer = shap.GradientExplainer(model, [bg_seq, bg_static])
    sv = explainer.shap_values(
        [seq.astype(np.float32), static.astype(np.float32)]
    )
    n_seq_flat = seq.shape[1] * seq.shape[2]
    seq_cols  = [f"seq_{i}" for i in range(n_seq_flat)]
    stat_cols = ["area_m2", "cad_con_ra", "lat", "lon"]
    sv_seq_flat = np.array(sv[0]).reshape(len(seq), -1)
    sv_static   = np.array(sv[1]).reshape(len(seq), -1)
    df = pd.DataFrame(np.concatenate([sv_seq_flat, sv_static], axis=1),
                      columns=seq_cols + stat_cols)
    df.to_csv(os.path.join(RESULT_DIR, "shap_lstm.csv"), index=False)
    print("LSTM SHAP 저장 완료.")
    return sv


# ─── 메인 ────────────────────────────────────────────────────────────────────

def evaluate(parcel_df, county_df):
    df = load_dataset(parcel_df)
    _, test_df = split_train_test(df)
    scaler_X, scaler_y = load_scalers()

    all_metrics = []
    all_preds   = {}

    # XGBoost
    pred_xgb, true_xgb, xgb_model = predict_xgboost(test_df, scaler_X, scaler_y)
    all_metrics.append(_metrics(true_xgb, pred_xgb, "XGBoost"))
    all_preds["xgboost"] = pred_xgb

    # LSTM
    pred_lstm, true_lstm, lstm_model = predict_lstm(test_df, scaler_X, scaler_y)
    all_metrics.append(_metrics(true_lstm, pred_lstm, "LSTM"))
    all_preds["lstm"] = pred_lstm

    # SARIMAX
    pred_sar, true_sar = predict_sarimax(test_df, county_df)
    all_metrics.append(_metrics(true_sar, pred_sar, "SARIMAX"))
    all_preds["sarimax"] = pred_sar

    # Prophet
    pred_pro, true_pro = predict_prophet(test_df, county_df)
    all_metrics.append(_metrics(true_pro, pred_pro, "Prophet"))
    all_preds["prophet"] = pred_pro

    # 예측/메트릭 저장 (SHAP 이전에 먼저)
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(RESULT_DIR, "metrics", "test_metrics.csv"),
                      index=False, encoding="utf-8-sig")

    pred_df = test_df[["uid", "year", "area_m2", TARGET]].copy().reset_index(drop=True)
    for name, arr in all_preds.items():
        pred_df[f"pred_{name}"] = arr
    pred_df.to_csv(os.path.join(RESULT_DIR, "predictions_test.csv"),
                   index=False, encoding="utf-8-sig")

    # SHAP
    run_shap_xgboost(xgb_model, test_df, scaler_X)
    run_shap_lstm(lstm_model, test_df, scaler_X)

    print("\n=== 최종 테스트 메트릭 ===")
    print(metrics_df.to_string(index=False))
    return metrics_df, pred_df


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
    evaluate(pf, county_df)
