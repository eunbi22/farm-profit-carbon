"""
SHAP 분석 시각화.
  - XGBoost: summary plot (beeswarm), bar plot (mean |SHAP|)
  - LSTM: mean |SHAP| bar plot

저장 위치: train/result/
실행: python plot/plot_shap.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from config import RESULT_DIR
from dataset import ALL_FEATURES, GROWING_MONTHS

from _font import setup as _setup_font
_setup_font()
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

FEATURE_KOR = {
    "area_m2":        "필지면적 (m²)",
    "cad_con_ra":     "경작일치율 (%)",
    "lat":            "위도",
    "lon":            "경도",
    "ta_season_mean": "영농기 평균기온",
    "rn_season_sum":  "영농기 누적강수",
    **{f"ta_mean_m{m}": f"{m}월 평균기온" for m in GROWING_MONTHS},
    **{f"rn_sum_m{m}":  f"{m}월 누적강수"  for m in GROWING_MONTHS},
}


def _feature_labels(cols):
    return [FEATURE_KOR.get(c, c) for c in cols]


def plot_xgboost_shap():
    path = os.path.join(RESULT_DIR, "shap_xgboost.csv")
    if not os.path.exists(path):
        print("XGBoost SHAP 파일 없음, 스킵")
        return
    sv = pd.read_csv(path).values
    labels = _feature_labels(ALL_FEATURES)

    # beeswarm summary
    fig, ax = plt.subplots(figsize=(8, 7))
    shap.summary_plot(sv, feature_names=labels, show=False, plot_size=None)
    plt.title("XGBoost SHAP Summary (Beeswarm)", fontsize=12)
    plt.tight_layout()
    out = os.path.join(RESULT_DIR, "shap_xgboost_beeswarm.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"저장: {out}")

    # mean |SHAP| bar
    mean_abs = np.abs(sv).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1]
    fig, ax  = plt.subplots(figsize=(8, 6))
    ax.barh(
        [labels[i] for i in order],
        mean_abs[order],
        color="#4C72B0",
    )
    ax.invert_yaxis()
    ax.set_xlabel("평균 |SHAP 값|")
    ax.set_title("XGBoost – 변수 중요도 (SHAP)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULT_DIR, "shap_xgboost_bar.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"저장: {out}")


def plot_lstm_shap():
    path = os.path.join(RESULT_DIR, "shap_lstm.csv")
    if not os.path.exists(path):
        print("LSTM SHAP 파일 없음, 스킵")
        return
    sv_df = pd.read_csv(path)

    # 시퀀스 SHAP은 월별로 집계 (seq_0~9 → m5~m9 ta/rn)
    seq_cols  = [c for c in sv_df.columns if c.startswith("seq_")]
    stat_cols = ["area_m2", "cad_con_ra", "lat", "lon"]

    # 시퀀스: 앞 5개 = ta (May-Sep), 뒤 5개 = rn (May-Sep)
    n_months = len(GROWING_MONTHS)
    seq_vals  = sv_df[seq_cols].values  # (N, 10)
    ta_shap   = seq_vals[:, :n_months].mean(axis=0)    # (5,) 월별 ta
    rn_shap   = seq_vals[:, n_months:].mean(axis=0)    # (5,) 월별 rn
    stat_shap = np.abs(sv_df[stat_cols].values).mean(axis=0)

    # bar plot
    labels_seq = (
        [f"{m}월 평균기온" for m in GROWING_MONTHS] +
        [f"{m}월 누적강수" for m in GROWING_MONTHS]
    )
    vals_seq = list(np.abs(ta_shap)) + list(np.abs(rn_shap))
    labels_all = labels_seq + _feature_labels(stat_cols)
    vals_all   = vals_seq   + list(stat_shap)

    order = np.argsort(vals_all)[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(
        [labels_all[i] for i in order],
        [vals_all[i]   for i in order],
        color="#4C72B0",
    )
    ax.invert_yaxis()
    ax.set_xlabel("평균 |SHAP 값|")
    ax.set_title("LSTM – 변수 중요도 (SHAP)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULT_DIR, "shap_lstm_bar.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"저장: {out}")


if __name__ == "__main__":
    plot_xgboost_shap()
    plot_lstm_shap()
