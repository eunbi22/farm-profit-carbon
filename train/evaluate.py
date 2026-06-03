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
import torch
import xgboost as xgb
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_error

from config import RESULT_DIR, TEST_YEARS_START, RANDOM_SEED
from dataset import (
    load_dataset, split_train_test, load_scalers,
    prepare_tabular, prepare_lstm_sequences, prepare_county_series,
    ALL_FEATURES, TARGET,
)
from train_lstm import YieldLSTM
from disaggregate import compute_parcel_weights

SAVE_DIR = os.path.join(RESULT_DIR, "models")
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    with open(os.path.join(RESULT_DIR, "metrics", "lstm_best_params.json")) as f:
        best_p = json.load(f)
    model = YieldLSTM(
        hidden_size=best_p["hidden_size"],
        num_layers=best_p["num_layers"],
        dropout=best_p["dropout"],
    ).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "lstm.pt"),
                                     map_location=DEVICE))
    model.eval()

    seq, static, y_s = prepare_lstm_sequences(test_df, scaler_X, scaler_y)
    with torch.no_grad():
        pred_s = model(
            torch.tensor(seq,    dtype=torch.float32).to(DEVICE),
            torch.tensor(static, dtype=torch.float32).to(DEVICE),
        ).cpu().numpy()
    return _inverse_y(pred_s, scaler_y), _inverse_y(y_s, scaler_y), model


def _disaggregate_county_pred(county_pred_kg, test_df, reg_info):
    """군 단위 예측값을 필지 가중치로 분배 → 필지별 yield_per_10a."""
    pf_w = compute_parcel_weights(
        test_df, reg_info["a"], reg_info["b"], reg_info["c"]
    )
    results = []
    for year, grp in pf_w.groupby("year"):
        if year not in county_pred_kg:
            continue
        w_sum = grp["weight"].sum()
        pred_kg = county_pred_kg[year] * grp["weight"] / w_sum
        pred_10a = pred_kg / (grp["area_m2"] / 1000.0)
        results.append(pred_10a.values)
    return np.concatenate(results)


def predict_sarimax(test_df, county_df, reg_info):
    with open(os.path.join(SAVE_DIR, "sarimax.pkl"), "rb") as f:
        saved = pickle.load(f)
    county_series = prepare_county_series(test_df, county_df)
    test_county   = county_series[county_series["year"] >= TEST_YEARS_START]
    exog = test_county[saved["exog_cols"]].astype(float).values
    pred_10a_arr = saved["model"].forecast(steps=len(test_county), exog=exog)
    county_pred_kg = {
        row["year"]: row_pred * (row["area_ha"] * 100)  # 10a당 × 10a수 = 총kg
        for (_, row), row_pred in zip(
            county_df[county_df["year"] >= TEST_YEARS_START].iterrows(),
            pred_10a_arr,
        )
    }
    pred_parcel = _disaggregate_county_pred(county_pred_kg, test_df, reg_info)
    true_parcel = test_df.sort_values("year")[TARGET].values
    return pred_parcel, true_parcel


def predict_prophet(test_df, county_df, reg_info):
    with open(os.path.join(SAVE_DIR, "prophet.pkl"), "rb") as f:
        saved = pickle.load(f)
    county_series = prepare_county_series(test_df, county_df)
    test_county   = county_series[county_series["year"] >= TEST_YEARS_START].copy()
    test_county["ds"] = pd.to_datetime(test_county["year"].astype(str) + "-01-01")
    pred_df = saved["model"].predict(test_county[["ds"] + saved["exog_cols"]])
    county_pred_kg = {
        int(row["ds"].year): row["yhat"] * (
            county_df.loc[county_df["year"] == int(row["ds"].year), "area_ha"].values[0] * 100
        )
        for _, row in pred_df.iterrows()
    }
    pred_parcel = _disaggregate_county_pred(county_pred_kg, test_df, reg_info)
    true_parcel = test_df.sort_values("year")[TARGET].values
    return pred_parcel, true_parcel


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

    class LSTMWrapper(torch.nn.Module):
        """SHAP GradientExplainer용: 입력을 하나의 텐서로 flatten."""
        def __init__(self, base, n_seq_flat):
            super().__init__()
            self.base = base
            self.n_seq_flat = n_seq_flat

        def forward(self, x):
            seq_   = x[:, :self.n_seq_flat].reshape(-1, 5, 2)
            static_ = x[:, self.n_seq_flat:]
            return self.base(seq_, static_)

    n_seq_flat = seq.shape[1] * seq.shape[2]
    X_flat = np.concatenate([seq.reshape(len(seq), -1), static], axis=1)
    bg_flat = torch.tensor(X_flat[idx], dtype=torch.float32).to(DEVICE)

    wrapper = LSTMWrapper(model, n_seq_flat).to(DEVICE)
    explainer = shap.GradientExplainer(wrapper, bg_flat)
    sv_flat = explainer.shap_values(
        torch.tensor(X_flat, dtype=torch.float32).to(DEVICE)
    )
    # 앞 n_seq_flat 컬럼은 시퀀스 feature, 뒤 4는 static
    seq_cols  = [f"seq_{i}" for i in range(n_seq_flat)]
    stat_cols = ["area_m2", "cad_con_ra", "lat", "lon"]
    df = pd.DataFrame(sv_flat, columns=seq_cols + stat_cols)
    df.to_csv(os.path.join(RESULT_DIR, "shap_lstm.csv"), index=False)
    print("LSTM SHAP 저장 완료.")
    return sv_flat


# ─── 메인 ────────────────────────────────────────────────────────────────────

def evaluate(parcel_df, county_df, reg_info):
    df       = load_dataset(parcel_df)
    _, test_df = split_train_test(df)
    scaler_X, scaler_y = load_scalers()

    all_metrics = []
    all_preds   = {}

    # XGBoost
    pred_xgb, true_xgb, xgb_model = predict_xgboost(test_df, scaler_X, scaler_y)
    all_metrics.append(_metrics(true_xgb, pred_xgb, "XGBoost"))
    all_preds["xgboost"] = pred_xgb
    run_shap_xgboost(xgb_model, test_df, scaler_X)

    # LSTM
    pred_lstm, true_lstm, lstm_model = predict_lstm(test_df, scaler_X, scaler_y)
    all_metrics.append(_metrics(true_lstm, pred_lstm, "LSTM"))
    all_preds["lstm"] = pred_lstm
    run_shap_lstm(lstm_model, test_df, scaler_X)

    # SARIMAX
    pred_sar, true_sar = predict_sarimax(test_df, county_df, reg_info)
    all_metrics.append(_metrics(true_sar, pred_sar, "SARIMAX"))
    all_preds["sarimax"] = pred_sar

    # Prophet
    pred_pro, true_pro = predict_prophet(test_df, county_df, reg_info)
    all_metrics.append(_metrics(true_pro, pred_pro, "Prophet"))
    all_preds["prophet"] = pred_pro

    # 저장
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(RESULT_DIR, "metrics", "test_metrics.csv"),
                      index=False, encoding="utf-8-sig")

    pred_df = test_df[["uid", "year", TARGET]].copy().reset_index(drop=True)
    for name, arr in all_preds.items():
        pred_df[f"pred_{name}"] = arr
    pred_df.to_csv(os.path.join(RESULT_DIR, "predictions_test.csv"),
                   index=False, encoding="utf-8-sig")

    print("\n=== 최종 테스트 메트릭 ===")
    print(metrics_df.to_string(index=False))
    return metrics_df, pred_df


if __name__ == "__main__":
    from features import build_parcel_features
    from disaggregate import (
        load_county_production, fit_county_model,
        compute_parcel_weights, disaggregate_yield,
    )

    pf        = build_parcel_features()
    county_df = load_county_production()
    cw        = pf.groupby("year")[["ta_season_mean", "rn_season_sum"]].mean().reset_index()
    reg_info  = fit_county_model(county_df, cw)
    pf_w      = compute_parcel_weights(pf, reg_info["a"], reg_info["b"], reg_info["c"])
    parcel_df = disaggregate_yield(pf_w, county_df)
    evaluate(parcel_df, county_df, reg_info)
