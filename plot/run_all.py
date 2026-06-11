"""
모든 plot을 순서대로 실행.

실행: python plot/run_all.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train"))

from config import RESULT_DIR

def _section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)


# ── 1. Loss 곡선 (train 후 바로 가능) ────────────────
_section("Loss 곡선")
from plot_loss import plot_lstm_loss, plot_optuna_loss

plot_lstm_loss()
for m in ("xgboost", "sarimax", "prophet"):
    plot_optuna_loss(m)


# ── evaluate.py 결과가 있을 때만 나머지 실행 ──────────
predictions_path = os.path.join(RESULT_DIR, "predictions_test.csv")
metrics_path     = os.path.join(RESULT_DIR, "metrics", "test_metrics.csv")

if not os.path.exists(predictions_path):
    print(f"\n[SKIP] predictions_test.csv 없음 → evaluate.py 먼저 실행하세요.")
    print("  나머지 plot (trend / heatmap / bar / shap) 은 스킵합니다.")
    sys.exit(0)


# ── 2. Trend / 산점도 ─────────────────────────────────
_section("Trend & 산점도")
from plot_trend import load_predictions, plot_county_trend, plot_scatter

pred_df = load_predictions()
plot_county_trend(pred_df)
plot_scatter(pred_df)


# ── 3. Heatmap ───────────────────────────────────────
_section("Heatmap")
from plot_heatmap import load_geodata, plot_all_heatmaps, plot_error_heatmap

gdf = load_geodata()
plot_all_heatmaps(gdf, pred_df)
plot_error_heatmap(gdf, pred_df, model="xgboost")


# ── 4. Bar 그래프 ─────────────────────────────────────
_section("Bar 그래프")
from plot_bar import load_metrics, plot_model_comparison, plot_yearly_county, plot_area_bin_yield

metrics_df = load_metrics()
plot_model_comparison(metrics_df)
plot_yearly_county(pred_df)
plot_area_bin_yield(pred_df)


# ── 5. SHAP ──────────────────────────────────────────
_section("SHAP")
from plot_shap import plot_xgboost_shap, plot_lstm_shap

plot_xgboost_shap()
plot_lstm_shap()


print(f"\n모든 plot 완료. 저장 위치: {RESULT_DIR}")
