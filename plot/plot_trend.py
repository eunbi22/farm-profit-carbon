"""
연도별 실측 vs 예측 생산량 trend 시각화.
  - 군 단위 집계값 비교 (4개 모델 오버레이)
  - 산점도 (실측 vs 예측, 필지 단위)

저장 위치: train/result/
실행: python plot/plot_trend.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from config import RESULT_DIR

from _font import setup as _setup_font
_setup_font()
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

MODEL_COLORS = {
    "xgboost": "#DD8452",
    "lstm":    "#4C72B0",
    "sarimax": "#55A868",
    "prophet": "#C44E52",
}
MODEL_LABELS = {
    "xgboost": "XGBoost",
    "lstm":    "LSTM",
    "sarimax": "SARIMAX",
    "prophet": "Prophet",
}


def load_predictions():
    path = os.path.join(RESULT_DIR, "predictions_test.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"예측 결과 없음: {path}\n먼저 evaluate.py를 실행하세요.")
    return pd.read_csv(path, encoding="utf-8-sig")


def plot_county_trend(pred_df: pd.DataFrame):
    """연도별 군 평균 yield_per_10a 비교."""
    agg = pred_df.groupby("year").mean(numeric_only=True).reset_index()
    models = [c.replace("pred_", "") for c in pred_df.columns if c.startswith("pred_")]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(agg["year"], agg["yield_per_10a"],
            "ko-", linewidth=2, markersize=7, label="실측", zorder=5)
    for m in models:
        col = f"pred_{m}"
        if col in agg:
            ax.plot(agg["year"], agg[col],
                    color=MODEL_COLORS.get(m, "gray"),
                    linestyle="--", linewidth=1.6, marker="s",
                    markersize=5, label=MODEL_LABELS.get(m, m))

    years = agg["year"].tolist()
    ax.set_xticks(years)
    ax.set_xticklabels([str(y) for y in years])
    if len(years) == 1:
        margin = 0.5
        ax.set_xlim(years[0] - margin, years[0] + margin)
    ax.set_title("군 평균 10a당 생산량 – 실측 vs 예측 (테스트셋)")
    ax.set_xlabel("연도")
    ax.set_ylabel("10a당 생산량 (kg)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULT_DIR, "trend_county.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"저장: {out}")


def plot_scatter(pred_df: pd.DataFrame):
    """필지별 실측 vs 예측 산점도 (모델별 subplot)."""
    models = [c.replace("pred_", "") for c in pred_df.columns if c.startswith("pred_")]
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    y_true = pred_df["yield_per_10a"].values
    lim_min = y_true.min() * 0.9
    lim_max = y_true.max() * 1.1

    for ax, m in zip(axes, models):
        col = f"pred_{m}"
        if col not in pred_df:
            continue
        y_pred = pred_df[col].values
        ax.scatter(y_true, y_pred, s=3, alpha=0.3,
                   color=MODEL_COLORS.get(m, "gray"))
        ax.plot([lim_min, lim_max], [lim_min, lim_max],
                "k--", linewidth=1, label="y=x")
        ax.set_xlim(lim_min, lim_max)
        ax.set_ylim(lim_min, lim_max)
        ax.set_title(MODEL_LABELS.get(m, m))
        ax.set_xlabel("실측 (kg/10a)")
        if ax == axes[0]:
            ax.set_ylabel("예측 (kg/10a)")
        ax.grid(True, alpha=0.2)

    fig.suptitle("필지별 실측 vs 예측 (테스트셋)", fontsize=13)
    plt.tight_layout()
    out = os.path.join(RESULT_DIR, "scatter_parcel.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"저장: {out}")


if __name__ == "__main__":
    pred_df = load_predictions()
    plot_county_trend(pred_df)
    plot_scatter(pred_df)
