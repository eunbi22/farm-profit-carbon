"""
EDA 전체 실행 통합 스크립트.

데이터 소스:
  - haenam_merged.csv       : 2020-05-02 ~ (일별 기온/강수/생산량) → 시계열/분포 분석
  - haenam_weather_grid.csv : 2023-01-01 ~ 2024-12-13 격자 데이터   → 공간 heatmap
  - rice_daily_price_*.csv  : 2024~ 쌀 도매가격

실행: python eda/run_all.py
"""

import sys, os, warnings
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "plot"))

from pathlib import Path
from _font import setup as _setup_font
_setup_font()

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

DATA_DIR = Path(__file__).parent.parent / "data"
SAVE_DIR  = Path(__file__).parent / "plots"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def _section(title):
    print(f"\n{'='*50}\n  {title}\n{'='*50}")


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"저장: {path}")


# ── 공통 데이터 로딩 ──────────────────────────────────
merged_path = DATA_DIR / "haenam_merged.csv"
if not merged_path.exists():
    print(f"[ERROR] {merged_path} 없음 — 종료")
    sys.exit(1)

print("haenam_merged.csv 로딩...")
df = pd.read_csv(merged_path)
df["date"]    = pd.to_datetime(df["date"])
df["year"]    = df["date"].dt.year
df["month"]   = df["date"].dt.month
df["ta"]      = pd.to_numeric(df["temp_avg"],    errors="coerce")
df["rn_day"]  = pd.to_numeric(df["rainfall_mm"], errors="coerce")
print(f"  {len(df):,}행  {df['date'].min().date()} ~ {df['date'].max().date()}")
print(f"  포함 연도: {sorted(df['year'].unique())}")

YEARS = sorted(df["year"].unique())


# ── 1. 기상 시계열 ────────────────────────────────────
_section("1. 기상 시계열 (2020~)")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
fig.suptitle(f"해남군 일별 기상 시계열 ({df['date'].min().year}~{df['date'].max().year})", fontsize=14)

ax1.plot(df["date"], df["ta"], linewidth=0.7, color="#d7191c", label="일평균기온")
ax1.axhline(df["ta"].mean(), color="black", linewidth=0.8, linestyle="--",
            alpha=0.5, label=f"전체 평균 {df['ta'].mean():.1f}°C")
ax1.set_ylabel("기온 (°C)")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

ax2.bar(df["date"], df["rn_day"], width=1.0, color="#2c7bb6", alpha=0.8, label="강수량")
ax2.set_ylabel("강수량 (mm)")
ax2.legend(fontsize=9)
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
ax2.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
_save(fig, SAVE_DIR / "01_weather_timeseries.png")


# ── 2. 연도별 박스플롯 ────────────────────────────────
_section("2. 연도별 기상 분포")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("연도별 기상 분포 (Box Plot)", fontsize=13)

ta_by_year = [df.loc[df["year"] == y, "ta"].dropna().values for y in YEARS]
rn_by_year = [df.loc[df["year"] == y, "rn_day"].dropna().values for y in YEARS]

bp_kw = dict(patch_artist=True,
             boxprops=dict(facecolor="#a6cee3", color="#1f78b4"),
             medianprops=dict(color="#e31a1c", linewidth=2))
ax1.boxplot(ta_by_year, labels=YEARS, **bp_kw)
ax1.set_title("일평균 기온 (°C)")
ax1.set_ylabel("기온 (°C)")
ax1.grid(True, alpha=0.3, axis="y")

rn_log = [np.log10(v[v > 0]) if (v > 0).any() else np.array([np.nan]) for v in rn_by_year]
rn_kw  = dict(patch_artist=True,
              boxprops=dict(facecolor="#fed9a6", color="#e6550d"),
              medianprops=dict(color="#e31a1c", linewidth=2))
ax2.boxplot(rn_log, labels=YEARS, **rn_kw)
ax2.set_title("일강수량 (강수일만, log₁₀)")
ax2.set_ylabel("log₁₀(강수량) [mm]")
ticks = [0, 0.5, 1, 1.5, 2, 2.5]
ax2.set_yticks(ticks)
ax2.set_yticklabels([f"$10^{{{v}}}$ ({10**v:.0f}mm)" for v in ticks], fontsize=8)
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
_save(fig, SAVE_DIR / "02_weather_boxplot_by_year.png")


# ── 3. 월별 기상 히트맵 ──────────────────────────────
_section("3. 연도×월별 기상 패턴")

ta_pivot  = df.groupby(["year", "month"])["ta"].mean().unstack()
rn_pivot  = df.groupby(["year", "month"])["rn_day"].sum().unstack()
month_lbl = ["1월","2월","3월","4월","5월","6월","7월","8월","9월","10월","11월","12월"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7))
fig.suptitle("연도×월별 기상 패턴", fontsize=13)

sns.heatmap(ta_pivot, ax=ax1, cmap="RdYlBu_r", annot=True, fmt=".1f",
            xticklabels=month_lbl, linewidths=0.5,
            cbar_kws={"label": "°C"}, annot_kws={"size": 9})
ax1.set_title("월평균 기온 (°C)")
ax1.set_xlabel("")
ax1.set_ylabel("연도")

sns.heatmap(rn_pivot, ax=ax2, cmap="Blues", annot=True, fmt=".0f",
            xticklabels=month_lbl, linewidths=0.5,
            cbar_kws={"label": "mm"}, annot_kws={"size": 9})
ax2.set_title("월별 누적 강수량 (mm)")
ax2.set_xlabel("월")
ax2.set_ylabel("연도")

plt.tight_layout()
_save(fig, SAVE_DIR / "03_monthly_weather_heatmap.png")


# ── 4. 영농기 연도별 요약 ─────────────────────────────
_section("4. 영농기(5~9월) 연도별 요약")

gs = df[df["month"].isin([5, 6, 7, 8, 9])]
yearly = gs.groupby("year").agg(ta_mean=("ta", "mean"), rn_sum=("rn_day", "sum")).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("영농기(5~9월) 연도별 기상 요약", fontsize=13)

bars = axes[0].bar(yearly["year"].astype(str), yearly["ta_mean"], color="#d7191c", edgecolor="white")
axes[0].set_title("영농기 평균기온 (°C)")
axes[0].set_ylabel("평균기온 (°C)")
axes[0].grid(True, axis="y", alpha=0.3)
for bar, val in zip(bars, yearly["ta_mean"]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=9)

bars = axes[1].bar(yearly["year"].astype(str), yearly["rn_sum"], color="#2c7bb6", edgecolor="white")
axes[1].set_title("영농기 누적강수량 (mm)")
axes[1].set_ylabel("누적강수량 (mm)")
axes[1].grid(True, axis="y", alpha=0.3)
for bar, val in zip(bars, yearly["rn_sum"]):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                 f"{val:.0f}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
_save(fig, SAVE_DIR / "04_growing_season_summary.png")


# ── 5. 기온 계절분해 ──────────────────────────────────
_section("5. 월평균 기온 계절분해")

try:
    monthly_ta = df.set_index("date")["ta"].resample("MS").mean().dropna().asfreq("MS").interpolate()
    result = seasonal_decompose(monthly_ta, model="additive", period=12, extrapolate_trend="freq")

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    titles     = ["원본 시계열", "추세 (Trend)", "계절성 (Seasonal)", "잔차 (Residual)"]
    components = [result.observed, result.trend, result.seasonal, result.resid]
    colors     = ["#2c7bb6", "#1a9641", "#d7191c", "#888888"]
    for ax, title, comp, color in zip(axes, titles, components, colors):
        ax.plot(comp.index, comp.values, linewidth=1.3, color=color)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("°C")
        ax.grid(True, alpha=0.3)
    fig.suptitle("해남군 월평균 기온 계절분해", fontsize=13)
    plt.tight_layout()
    _save(fig, SAVE_DIR / "05_seasonal_decomposition.png")
except Exception as e:
    print(f"[SKIP] 계절분해: {e}")


# ── 6. ACF / PACF ─────────────────────────────────────
_section("6. ACF / PACF")

try:
    monthly_ta = df.set_index("date")["ta"].resample("MS").mean().dropna().asfreq("MS").interpolate()
    max_lags = min(24, len(monthly_ta) // 2 - 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
    plot_acf(monthly_ta,  lags=max_lags, ax=ax1, alpha=0.05, title="자기상관함수 (ACF) — 월평균 기온")
    plot_pacf(monthly_ta, lags=max_lags, ax=ax2, alpha=0.05, method="ywm", title="편자기상관함수 (PACF) — 월평균 기온")
    for ax in (ax1, ax2):
        ax.set_xlabel("Lag (월)")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, SAVE_DIR / "06_acf_pacf.png")
except Exception as e:
    print(f"[SKIP] ACF/PACF: {e}")


# ── 7. 공간 기상 heatmap (격자 데이터) ───────────────
_section("7. 공간 기상 분포 (격자 데이터, 2023~2024)")

grid_path = DATA_DIR / "haenam_weather_grid.csv"
if not grid_path.exists():
    print(f"[SKIP] {grid_path.name} 없음")
else:
    from eda_farm import _daily_mean, plot_spatial_heatmap
    print("haenam_weather_grid.csv 로딩 중 (대용량)...")
    weather = pd.read_csv(grid_path)
    weather["date"] = pd.to_datetime(weather["date"].astype(str), format="%Y%m%d")
    weather["ta"]   = weather["ta"] / 10
    plot_spatial_heatmap(weather, SAVE_DIR)


# ── 8. 쌀 가격 분석 ──────────────────────────────────
_section("8. 쌀 도매가격 분석")

price_path = DATA_DIR / "rice_daily_price_2020_2025.csv"
if not price_path.exists():
    print(f"[SKIP] {price_path.name} 없음")
else:
    price_df = pd.read_csv(price_path)
    price_df["date"] = pd.to_datetime(
        price_df["year"].astype(str) + "/" + price_df["date"].astype(str),
        format="%Y/%m/%d", errors="coerce"
    )
    price_df = price_df.dropna(subset=["date", "price"]).sort_values("date").reset_index(drop=True)
    price_df["ma30"]  = price_df["price"].rolling(30, center=True, min_periods=1).mean()
    price_df["year"]  = price_df["date"].dt.year
    price_df["month"] = price_df["date"].dt.month

    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    fig.suptitle("쌀 20kg 도매가격 분석", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(price_df["date"], price_df["price"], color="steelblue", linewidth=0.8, alpha=0.6, label="일별 가격")
    ax.plot(price_df["date"], price_df["ma30"],  color="crimson",   linewidth=2.0, label="30일 이동평균")
    ax.set_ylabel("가격 (원/20kg)")
    ax.set_title("전체 기간 가격 추이")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    yearly_p = price_df.groupby("year")["price"].agg(["mean", "min", "max"]).reset_index()
    bars = ax.bar(yearly_p["year"].astype(str), yearly_p["mean"], color="steelblue", edgecolor="white", zorder=3)
    ax.errorbar(range(len(yearly_p)), yearly_p["mean"],
                yerr=[yearly_p["mean"] - yearly_p["min"], yearly_p["max"] - yearly_p["mean"]],
                fmt="none", color="gray", capsize=5, linewidth=1.2, zorder=4)
    for bar, val in zip(bars, yearly_p["mean"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f"{int(val):,}원", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("평균 가격 (원/20kg)")
    ax.set_title("연도별 평균 가격")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[2]
    p_years = sorted(price_df["year"].unique())
    cmap = plt.cm.get_cmap("tab10", len(p_years))
    for i, yr in enumerate(p_years):
        monthly_avg = (
            price_df[price_df["year"] == yr]
            .groupby("month")["price"].mean()
            .reindex(range(1, 13))
        )
        ax.plot(monthly_avg.index, monthly_avg.values,
                marker="o", markersize=4, linewidth=1.6, color=cmap(i), label=str(yr))
    ax.set_xlabel("월")
    ax.set_ylabel("월평균 가격 (원/20kg)")
    ax.set_title("월별 계절성 패턴 (연도별)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([f"{m}월" for m in range(1, 13)])
    ax.legend(title="연도", bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save(fig, SAVE_DIR / "08_price_analysis.png")


print(f"\n모든 EDA 완료. 저장 위치: {SAVE_DIR}")
