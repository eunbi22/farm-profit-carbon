"""
필지별 예측 생산량 공간 heatmap (Choropleth).
  - 각 모델별 PNG
  - 실측 vs XGBoost 비교 패널

저장 위치: train/result/
실행: python plot/plot_heatmap.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from config import RESULT_DIR, FARMMAP_GEO

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

MODEL_LABELS = {
    "xgboost": "XGBoost",
    "lstm":    "LSTM",
    "sarimax": "SARIMAX",
    "prophet": "Prophet",
}


def load_geodata():
    print("geojson 로딩 (heatmap용)…")
    gdf = gpd.read_file(FARMMAP_GEO)
    gdf["uid"] = gdf["uid"].astype(str)
    return gdf


def load_predictions():
    path = os.path.join(RESULT_DIR, "predictions_test.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"예측 결과 없음: {path}")
    return pd.read_csv(path, encoding="utf-8-sig", dtype={"uid": str})


def _single_heatmap(gdf_merged, col, title, ax, vmin, vmax, cmap="YlOrRd"):
    gdf_merged.plot(
        column=col, ax=ax, cmap=cmap,
        vmin=vmin, vmax=vmax,
        missing_kwds={"color": "lightgrey", "label": "데이터 없음"},
        linewidth=0.05, edgecolor="white",
    )
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="kg/10a", shrink=0.7, pad=0.02)
    ax.set_title(title, fontsize=11)
    ax.axis("off")


def plot_all_heatmaps(gdf, pred_df):
    models = [c.replace("pred_", "") for c in pred_df.columns if c.startswith("pred_")]
    years  = sorted(pred_df["year"].unique())

    for year in years:
        yr_df = pred_df[pred_df["year"] == year]

        # geojson 병합
        merged = gdf.merge(yr_df, on="uid", how="left")

        # 공통 color range
        vals = [merged["yield_per_10a"].dropna().values]
        for m in models:
            c = f"pred_{m}"
            if c in merged:
                vals.append(merged[c].dropna().values)
        all_vals = np.concatenate(vals)
        vmin, vmax = np.percentile(all_vals, 2), np.percentile(all_vals, 98)

        n = 1 + len(models)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 6))
        if n == 1:
            axes = [axes]

        _single_heatmap(merged, "yield_per_10a", f"실측 ({year}년)",
                        axes[0], vmin, vmax)
        for ax, m in zip(axes[1:], models):
            col = f"pred_{m}"
            if col in merged:
                _single_heatmap(merged, col,
                                f"{MODEL_LABELS.get(m, m)} 예측 ({year}년)",
                                ax, vmin, vmax)

        plt.suptitle(f"필지별 10a당 생산량 heatmap – {year}년", fontsize=13, y=1.01)
        plt.tight_layout()
        out = os.path.join(RESULT_DIR, f"heatmap_{year}.png")
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"저장: {out}")


def plot_error_heatmap(gdf, pred_df, model="xgboost"):
    """예측 오차(pred - actual) 공간 분포."""
    col = f"pred_{model}"
    if col not in pred_df.columns:
        return
    years = sorted(pred_df["year"].unique())
    for year in years:
        yr_df = pred_df[pred_df["year"] == year].copy()
        yr_df["error"] = yr_df[col] - yr_df["yield_per_10a"]
        merged = gdf.merge(yr_df[["uid", "error"]], on="uid", how="left")

        lim = np.percentile(merged["error"].dropna().abs(), 98)
        fig, ax = plt.subplots(figsize=(8, 7))
        _single_heatmap(merged, "error",
                        f"{MODEL_LABELS.get(model, model)} 예측 오차 ({year}년)",
                        ax, -lim, lim, cmap="RdBu_r")
        plt.tight_layout()
        out = os.path.join(RESULT_DIR, f"heatmap_error_{model}_{year}.png")
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        print(f"저장: {out}")


if __name__ == "__main__":
    gdf     = load_geodata()
    pred_df = load_predictions()
    plot_all_heatmaps(gdf, pred_df)
    plot_error_heatmap(gdf, pred_df, model="xgboost")
