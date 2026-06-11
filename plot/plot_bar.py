"""
발표/보고서용 대표값 막대 그래프.
  1. 모델별 RMSE / MAE / MAPE 비교
  2. 연도별 군 총생산량 실측 vs 예측 비교
  3. 필지 면적 구간별 평균 예측 생산량 분포

저장 위치: train/result/
실행: python plot/plot_bar.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import RESULT_DIR

from _font import setup as _setup_font
_setup_font()
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})

MODEL_COLORS = {
    "XGBoost": "#DD8452",
    "LSTM":    "#4C72B0",
    "SARIMAX": "#55A868",
    "Prophet": "#C44E52",
}


def load_metrics():
    path = os.path.join(RESULT_DIR, "metrics", "test_metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"메트릭 파일 없음: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def load_predictions():
    path = os.path.join(RESULT_DIR, "predictions_test.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"예측 결과 없음: {path}")
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"uid": str})


def plot_model_comparison(metrics_df: pd.DataFrame):
    """모델별 RMSE, MAE, MAPE 그룹 막대 그래프."""
    metrics = ["RMSE", "MAE", "MAPE"]
    units   = ["kg/10a", "kg/10a", "%"]
    n_models = len(metrics_df)
    x = np.arange(n_models)
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, met, unit in zip(axes, metrics, units):
        vals   = metrics_df[met].values
        colors = [MODEL_COLORS.get(m, "steelblue") for m in metrics_df["model"]]
        bars   = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df["model"], rotation=15, ha="right")
        ax.set_ylabel(unit)
        ax.set_title(met)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=9)

    fig.suptitle("모델 성능 비교 (테스트셋)", fontsize=13)
    plt.tight_layout()
    out = os.path.join(RESULT_DIR, "bar_model_comparison.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"저장: {out}")


def plot_yearly_county(pred_df: pd.DataFrame):
    """연도별 군 평균 실측 vs 모델별 예측 막대 그래프."""
    models = [c.replace("pred_", "") for c in pred_df.columns if c.startswith("pred_")]
    agg    = pred_df.groupby("year").mean(numeric_only=True).reset_index()
    years  = agg["year"].values
    n_y    = len(years)
    n_cols = 1 + len(models)
    x      = np.arange(n_y)
    width  = 0.8 / n_cols

    fig, ax = plt.subplots(figsize=(10, 5))
    offsets = np.linspace(-(n_cols - 1) / 2, (n_cols - 1) / 2, n_cols) * width

    ax.bar(x + offsets[0], agg["yield_per_10a"], width,
           label="실측", color="dimgray", edgecolor="white")
    for i, m in enumerate(models):
        col = f"pred_{m}"
        if col in agg:
            ax.bar(x + offsets[i + 1], agg[col], width,
                   label=m.upper(),
                   color=MODEL_COLORS.get(m.upper(), "steelblue"),
                   edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_xlabel("연도")
    ax.set_ylabel("10a당 생산량 (kg)")
    ax.set_title("연도별 군 평균 생산량 – 실측 vs 예측")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULT_DIR, "bar_yearly_county.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"저장: {out}")


def plot_area_bin_yield(pred_df: pd.DataFrame):
    """필지 면적 구간별 평균 예측 생산량 분포 (XGBoost 기준)."""
    col = "pred_xgboost"
    if col not in pred_df.columns:
        print("XGBoost 예측값 없음, 스킵")
        return

    df = pred_df[["area_m2", "yield_per_10a", col]].dropna().copy()
    bins   = [0, 500, 1000, 2000, 5000, np.inf]
    labels = ["~500", "500~1000", "1000~2000", "2000~5000", "5000+"]
    df["area_bin"] = pd.cut(df["area_m2"], bins=bins, labels=labels)

    agg = df.groupby("area_bin", observed=True).agg(
        actual=("yield_per_10a", "mean"),
        predicted=(col, "mean"),
        count=("yield_per_10a", "count"),
    ).reset_index()

    x     = np.arange(len(agg))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, agg["actual"],    width, label="실측",           color="dimgray")
    ax.bar(x + width / 2, agg["predicted"], width, label="XGBoost 예측",   color=MODEL_COLORS["XGBoost"])

    ax2 = ax.twinx()
    ax2.plot(x, agg["count"], "o--", color="navy", linewidth=1.5,
             markersize=6, label="필지 수")
    ax2.set_ylabel("필지 수", color="navy")
    ax2.tick_params(axis="y", labelcolor="navy")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(m²)" for l in agg["area_bin"]])
    ax.set_xlabel("필지 면적 구간")
    ax.set_ylabel("평균 10a당 생산량 (kg)")
    ax.set_title("면적 구간별 평균 생산량 – 실측 vs XGBoost 예측")

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULT_DIR, "bar_area_bin_yield.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"저장: {out}")


if __name__ == "__main__":
    metrics_df = load_metrics()
    pred_df    = load_predictions()
    plot_model_comparison(metrics_df)
    plot_yearly_county(pred_df)
    plot_area_bin_yield(pred_df)
