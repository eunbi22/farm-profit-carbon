"""
EDA plots for farm-profit-carbon pipeline.

Data:
  1. data/haenam_weather_grid.csv  — daily grid weather (ta: 일평균기온, rn_day: 일강수량)
  2. data/rice_20kg_wholesale_price.csv — 20kg 쌀 도매가격
  3. data/haenam_merged.csv — 연도별 쌀 생산량 (production_ton)

Outputs saved to eda/plots/.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore")

matplotlib.rcParams["font.family"] = ["AppleGothic", "Malgun Gothic", "NanumGothic", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

DATA_DIR = Path(__file__).parent.parent / "data"
SAVE_DIR = Path(__file__).parent / "plots"


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA] Saved: {path}")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and preprocess all three datasets."""
    print("[EDA] Loading haenam_weather_grid.csv (large file, may take a moment)...")
    weather = pd.read_csv(DATA_DIR / "haenam_weather_grid.csv")
    weather["date"] = pd.to_datetime(weather["date"].astype(str), format="%Y%m%d")
    weather["ta"] = weather["ta"] / 10  # 0.1°C 단위 → °C

    print("[EDA] Loading rice_20kg_wholesale_price.csv...")
    price = pd.read_csv(DATA_DIR / "rice_20kg_wholesale_price.csv")
    price["date"] = pd.to_datetime(price["date"])

    print("[EDA] Loading haenam_merged.csv...")
    merged = pd.read_csv(DATA_DIR / "haenam_merged.csv")
    merged["date"] = pd.to_datetime(merged["date"])

    return weather, price, merged


def _daily_mean(weather: pd.DataFrame) -> pd.DataFrame:
    """Spatial mean of ta and rn_day per day across all grid cells."""
    return weather.groupby("date")[["ta", "rn_day"]].mean().sort_index()


def _monthly_mean(daily: pd.DataFrame) -> pd.DataFrame:
    return daily.resample("MS").mean()


def _annual_production(merged: pd.DataFrame) -> pd.Series:
    merged = merged.copy()
    merged["year"] = merged["date"].dt.year
    return merged.groupby("year")["production_ton"].first().sort_index()


# ──────────────────────────────────────────────
# Plot 01 — Weather time-series
# ──────────────────────────────────────────────

def plot_weather_timeseries(daily: pd.DataFrame, save_dir: Path = SAVE_DIR) -> None:
    """Daily spatial-mean ta and rn_day over the full period."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle("해남군 일평균 기온·강수량 시계열 (그리드 공간 평균)", fontsize=14)

    ax1.plot(daily.index, daily["ta"], linewidth=0.9, color="#d7191c", label="기온")
    ax1.set_ylabel("일평균 기온 (°C)", fontsize=10)
    ax1.set_title("일평균 기온 (공간 평균)", fontsize=11)
    ax1.axhline(daily["ta"].mean(), color="black", linewidth=0.8, linestyle="--", alpha=0.6, label=f"전체 평균 {daily['ta'].mean():.1f}°C")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.bar(daily.index, daily["rn_day"], width=1.5, color="#2c7bb6", alpha=0.75, label="강수량")
    ax2.set_ylabel("일강수량 (mm)", fontsize=10)
    ax2.set_title("일강수량 (공간 평균)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    _save(fig, save_dir / "01_weather_timeseries.png")


# ──────────────────────────────────────────────
# Plot 02 — Spatial grid heatmap
# ──────────────────────────────────────────────

def plot_spatial_heatmap(weather: pd.DataFrame, save_dir: Path = SAVE_DIR) -> None:
    """Mean ta and rn_day per grid cell (ny × nx heatmap)."""
    grid_mean = weather.groupby(["ny", "nx"])[["ta", "rn_day"]].mean().reset_index()
    ta_pivot   = grid_mean.pivot(index="ny", columns="nx", values="ta")
    rn_pivot   = grid_mean.pivot(index="ny", columns="nx", values="rn_day")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("해남군 격자별 기상 평균값 (2020–2025)", fontsize=14)

    im1 = ax1.imshow(ta_pivot.values, aspect="auto", cmap="RdYlBu_r", origin="lower")
    ax1.set_title("격자별 평균 기온 (°C)", fontsize=12)
    ax1.set_xlabel("nx (경도 격자)", fontsize=9)
    ax1.set_ylabel("ny (위도 격자)", fontsize=9)
    ax1.set_xticks(range(0, ta_pivot.shape[1], 10))
    ax1.set_xticklabels(ta_pivot.columns[::10], fontsize=7, rotation=45)
    ax1.set_yticks(range(0, ta_pivot.shape[0], 10))
    ax1.set_yticklabels(ta_pivot.index[::10], fontsize=7)
    plt.colorbar(im1, ax=ax1, label="°C")

    im2 = ax2.imshow(rn_pivot.values, aspect="auto", cmap="Blues", origin="lower")
    ax2.set_title("격자별 평균 강수량 (mm/일)", fontsize=12)
    ax2.set_xlabel("nx (경도 격자)", fontsize=9)
    ax2.set_ylabel("ny (위도 격자)", fontsize=9)
    ax2.set_xticks(range(0, rn_pivot.shape[1], 10))
    ax2.set_xticklabels(rn_pivot.columns[::10], fontsize=7, rotation=45)
    ax2.set_yticks(range(0, rn_pivot.shape[0], 10))
    ax2.set_yticklabels(rn_pivot.index[::10], fontsize=7)
    plt.colorbar(im2, ax=ax2, label="mm/일")

    fig.tight_layout()
    _save(fig, save_dir / "02_spatial_heatmap.png")


# ──────────────────────────────────────────────
# Plot 03 — Weather distributions
# ──────────────────────────────────────────────

def plot_weather_distributions(weather: pd.DataFrame, save_dir: Path = SAVE_DIR) -> None:
    """Histogram + KDE for ta and rn_day (all grid-cell observations)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("기상 변수 분포 (Histogram + KDE)", fontsize=13)

    ta_vals = weather["ta"].dropna()
    axes[0].hist(ta_vals, bins=60, density=True, alpha=0.55, color="#d7191c", edgecolor="white")
    ta_vals.plot.kde(ax=axes[0], color="#8b0000", linewidth=2)
    axes[0].set_title("일평균 기온 (°C)", fontsize=11)
    axes[0].set_xlabel("기온 (°C)", fontsize=9)
    axes[0].set_ylabel("밀도", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    rn_vals = weather["rn_day"].dropna()
    rn_nonzero = rn_vals[rn_vals > 0]
    log_rn = np.log10(rn_nonzero)
    axes[1].hist(log_rn, bins=60, density=True, alpha=0.55, color="#2c7bb6", edgecolor="white")
    log_rn.plot.kde(ax=axes[1], color="#08306b", linewidth=2)
    axes[1].set_title("일강수량 분포 (강수일만, log₁₀ 스케일)", fontsize=11)
    axes[1].set_xlabel("log₁₀(강수량) [mm]", fontsize=9)
    axes[1].set_ylabel("밀도", fontsize=9)
    tick_vals = [0, 0.5, 1, 1.5, 2, 2.5, 3]
    axes[1].set_xticks(tick_vals)
    axes[1].set_xticklabels([f"$10^{{{v}}}$\n({10**v:.0f}mm)" for v in tick_vals], fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, save_dir / "03_weather_distributions.png")


# ──────────────────────────────────────────────
# Plot 04 — Box-plots by year
# ──────────────────────────────────────────────

def plot_weather_boxplot_by_year(weather: pd.DataFrame, save_dir: Path = SAVE_DIR) -> None:
    """Yearly box-plots for ta and rn_day."""
    weather = weather.copy()
    weather["year"] = weather["date"].dt.year
    years = sorted(weather["year"].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("연도별 기상 분포 (Box Plot)", fontsize=13)

    ta_by_year  = [weather.loc[weather["year"] == y, "ta"].dropna().values for y in years]
    rn_by_year  = [weather.loc[weather["year"] == y, "rn_day"].dropna().values for y in years]

    bp_kw = dict(patch_artist=True,
                 boxprops=dict(facecolor="#a6cee3", color="#1f78b4"),
                 medianprops=dict(color="#e31a1c", linewidth=2))

    ax1.boxplot(ta_by_year, labels=years, **bp_kw)
    ax1.set_title("일평균 기온 (°C)", fontsize=11)
    ax1.set_ylabel("기온 (°C)", fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    # log10 변환 (강수일만, 0은 제외)
    rn_log_by_year = [np.log10(v[v > 0]) if (v > 0).any() else np.array([np.nan]) for v in rn_by_year]

    rn_bp_kw = dict(patch_artist=True,
                    boxprops=dict(facecolor="#fed9a6", color="#e6550d"),
                    medianprops=dict(color="#e31a1c", linewidth=2))
    ax2.boxplot(rn_log_by_year, labels=years, **rn_bp_kw)
    ax2.set_title("일강수량 (강수일만, log₁₀ 스케일)", fontsize=11)
    ax2.set_ylabel("log₁₀(강수량) [mm]", fontsize=9)
    tick_vals = [0, 0.5, 1, 1.5, 2, 2.5, 3]
    ax2.set_yticks(tick_vals)
    ax2.set_yticklabels([f"$10^{{{v}}}$ ({10**v:.0f}mm)" for v in tick_vals], fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    _save(fig, save_dir / "04_weather_boxplot_by_year.png")


# ──────────────────────────────────────────────
# Plot 05 — Seasonal decomposition of monthly ta
# ──────────────────────────────────────────────

def plot_seasonal_decomposition(daily: pd.DataFrame, save_dir: Path = SAVE_DIR) -> None:
    """Additive seasonal decomposition of monthly-mean temperature."""
    monthly_ta = daily["ta"].resample("MS").mean().dropna()
    monthly_ta = monthly_ta.asfreq("MS").interpolate()

    result = seasonal_decompose(monthly_ta, model="additive", period=12, extrapolate_trend="freq")

    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True)
    titles     = ["원본 시계열", "추세 (Trend)", "계절성 (Seasonal)", "잔차 (Residual)"]
    components = [result.observed, result.trend, result.seasonal, result.resid]
    colors     = ["#2c7bb6", "#1a9641", "#d7191c", "#888888"]

    for ax, title, comp, color in zip(axes, titles, components, colors):
        ax.plot(comp.index, comp.values, linewidth=1.3, color=color)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("°C", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("해남군 월평균 기온 계절분해 (Seasonal Decomposition)", fontsize=13)
    fig.tight_layout()
    _save(fig, save_dir / "05_seasonal_decomposition.png")


# ──────────────────────────────────────────────
# Plot 06 — ACF / PACF of monthly ta
# ──────────────────────────────────────────────

def plot_acf_pacf(daily: pd.DataFrame, save_dir: Path = SAVE_DIR, lags: int = 36) -> None:
    """ACF and PACF of monthly-mean temperature."""
    monthly_ta = daily["ta"].resample("MS").mean().dropna()
    monthly_ta = monthly_ta.asfreq("MS").interpolate()

    # PACF requires lags < 50% of sample size
    max_lags = min(lags, len(monthly_ta) // 2 - 1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
    plot_acf(monthly_ta,  lags=max_lags, ax=ax1, alpha=0.05, title="자기상관함수 (ACF) — 월평균 기온")
    plot_pacf(monthly_ta, lags=max_lags, ax=ax2, alpha=0.05, method="ywm", title="편자기상관함수 (PACF) — 월평균 기온")

    for ax in (ax1, ax2):
        ax.set_xlabel("Lag (월)", fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, save_dir / "06_acf_pacf.png")


# ──────────────────────────────────────────────
# Plot 07 — Rice price time-series
# ──────────────────────────────────────────────

def plot_rice_price_timeseries(price: pd.DataFrame, save_dir: Path = SAVE_DIR) -> None:
    """Rice price over time with monthly rolling average."""
    price_s = price.set_index("date")["price_krw_per_20kg"].sort_index()
    rolling = price_s.rolling(30, center=True, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(price_s.index, price_s.values, linewidth=0.8, color="#888888", alpha=0.6, label="일별 가격")
    ax.plot(rolling.index, rolling.values, linewidth=2.0, color="#d7191c", label="30일 이동평균")
    ax.set_title("쌀 20kg 도매가격 시계열 (2020–2026)", fontsize=13)
    ax.set_ylabel("가격 (원/20kg)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, save_dir / "07_rice_price_timeseries.png")


# ──────────────────────────────────────────────
# Plot 08 — Rice price distribution & change rate
# ──────────────────────────────────────────────

def plot_rice_price_distribution(price: pd.DataFrame, save_dir: Path = SAVE_DIR) -> None:
    """Price histogram + KDE, and daily change-rate distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("쌀 도매가격 분포", fontsize=13)

    p = price["price_krw_per_20kg"].dropna()
    axes[0].hist(p, bins=40, density=True, alpha=0.55, color="#2c7bb6", edgecolor="white")
    p.plot.kde(ax=axes[0], color="#08306b", linewidth=2)
    axes[0].set_title("가격 분포 (원/20kg)", fontsize=11)
    axes[0].set_xlabel("가격 (원)", fontsize=9)
    axes[0].set_ylabel("밀도", fontsize=9)
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    axes[0].grid(True, alpha=0.3)

    cr = price["daily_change_rate"].dropna()
    axes[1].hist(cr, bins=40, density=True, alpha=0.55, color="#fdae61", edgecolor="white")
    cr.plot.kde(ax=axes[1], color="#d7191c", linewidth=2)
    axes[1].set_title("일별 변동률 분포 (%)", fontsize=11)
    axes[1].set_xlabel("변동률 (%)", fontsize=9)
    axes[1].set_ylabel("밀도", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, save_dir / "08_rice_price_distribution.png")


# ──────────────────────────────────────────────
# Plot 09 — Annual rice production
# ──────────────────────────────────────────────

def plot_annual_production(merged: pd.DataFrame, save_dir: Path = SAVE_DIR) -> None:
    """Bar chart of annual rice production (production_ton)."""
    prod = _annual_production(merged)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(prod.index.astype(str), prod.values, color="#74c476", edgecolor="#238b45", linewidth=1.2)
    ax.set_title("해남군 연도별 쌀 생산량", fontsize=13)
    ax.set_xlabel("연도", fontsize=10)
    ax.set_ylabel("생산량 (톤)", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars, prod.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + prod.max() * 0.01,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ymin = prod.min() * 0.85
    ymax = prod.max() * 1.12
    ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    _save(fig, save_dir / "09_annual_production.png")


# ──────────────────────────────────────────────
# Plot 10 — Weather vs rice price correlation
# ──────────────────────────────────────────────

def plot_weather_price_correlation(
    daily: pd.DataFrame,
    price: pd.DataFrame,
    save_dir: Path = SAVE_DIR,
) -> None:
    """Monthly mean weather joined with monthly mean price — correlation heatmap + scatter."""
    monthly_w = daily.resample("MS").mean()
    monthly_p = (
        price.set_index("date")["price_krw_per_20kg"]
        .resample("MS").mean()
        .rename("price_krw_per_20kg")
    )
    combined = monthly_w.join(monthly_p, how="inner").dropna()
    combined.columns = ["월평균기온(°C)", "월평균강수량(mm)", "쌀가격(원/20kg)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("월별 기상·쌀가격 관계", fontsize=13)

    pairs = [
        ("월평균기온(°C)", "쌀가격(원/20kg)"),
        ("월평균강수량(mm)", "쌀가격(원/20kg)"),
        ("월평균기온(°C)", "월평균강수량(mm)"),
    ]
    for ax, (x_col, y_col) in zip(axes, pairs):
        ax.scatter(combined[x_col], combined[y_col], alpha=0.6, s=30, color="#2c7bb6", edgecolors="white", linewidth=0.4)
        r = combined[x_col].corr(combined[y_col])
        ax.set_xlabel(x_col, fontsize=9)
        ax.set_ylabel(y_col, fontsize=9)
        ax.set_title(f"r = {r:.3f}", fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, save_dir / "10_weather_price_scatter.png")

    corr = combined.corr(method="pearson")
    fig2, ax = plt.subplots(figsize=(6, 5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".3f", cmap="coolwarm",
                center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
                annot_kws={"size": 11})
    ax.set_title("기상·쌀가격 피어슨 상관관계", fontsize=12)
    fig2.tight_layout()
    _save(fig2, save_dir / "10_weather_price_correlation.png")


# ──────────────────────────────────────────────
# Plot 11 — Monthly mean weather overview
# ──────────────────────────────────────────────

def plot_monthly_weather_heatmap(daily: pd.DataFrame, save_dir: Path = SAVE_DIR) -> None:
    """Year × month heatmap for ta and rn_day (easier to read seasonality)."""
    df = daily.copy()
    df["year"]  = df.index.year
    df["month"] = df.index.month

    ta_pivot  = df.groupby(["year", "month"])["ta"].mean().unstack()
    rn_pivot  = df.groupby(["year", "month"])["rn_day"].sum().unstack()

    month_labels = ["1월", "2월", "3월", "4월", "5월", "6월",
                    "7월", "8월", "9월", "10월", "11월", "12월"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7))
    fig.suptitle("연도×월별 기상 패턴 (공간 평균)", fontsize=13)

    sns.heatmap(ta_pivot, ax=ax1, cmap="RdYlBu_r", annot=True, fmt=".1f",
                xticklabels=month_labels, linewidths=0.5,
                cbar_kws={"label": "°C"}, annot_kws={"size": 9})
    ax1.set_title("월평균 기온 (°C)", fontsize=11)
    ax1.set_xlabel("")
    ax1.set_ylabel("연도", fontsize=9)

    sns.heatmap(rn_pivot, ax=ax2, cmap="Blues", annot=True, fmt=".0f",
                xticklabels=month_labels, linewidths=0.5,
                cbar_kws={"label": "mm"}, annot_kws={"size": 9})
    ax2.set_title("월별 누적 강수량 (mm)", fontsize=11)
    ax2.set_xlabel("월", fontsize=9)
    ax2.set_ylabel("연도", fontsize=9)

    fig.tight_layout()
    _save(fig, save_dir / "11_monthly_weather_heatmap.png")


# ──────────────────────────────────────────────
# Plot 12 — Production vs weather & price summary
# ──────────────────────────────────────────────

def plot_production_vs_weather_price(
    daily: pd.DataFrame,
    price: pd.DataFrame,
    merged: pd.DataFrame,
    save_dir: Path = SAVE_DIR,
) -> None:
    """Annual production against growing-season weather and harvest-year price."""
    prod = _annual_production(merged)

    # Growing season: May–Sep
    daily_gs = daily[daily.index.month.isin([5, 6, 7, 8, 9])]
    annual_ta  = daily_gs.groupby(daily_gs.index.year)["ta"].mean().rename("생육기평균기온(°C)")
    annual_rn  = daily_gs.groupby(daily_gs.index.year)["rn_day"].sum().rename("생육기누적강수량(mm)")

    # Annual mean price
    annual_price = (
        price.set_index("date")["price_krw_per_20kg"]
        .resample("YS").mean()
        .rename("연간평균가격")
    )
    annual_price.index = annual_price.index.year

    years = sorted(set(prod.index) & set(annual_ta.index) & set(annual_price.index))
    df = pd.DataFrame({
        "생산량(톤)": prod[years],
        "생육기평균기온(°C)": annual_ta[years],
        "생육기누적강수량(mm)": annual_rn[years],
        "연간평균가격(원)": annual_price[years],
    })

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("연도별 생산량 vs 기상·가격 (생육기: 5–9월)", fontsize=13)

    pairs = [
        ("생육기평균기온(°C)", "생산량(톤)"),
        ("생육기누적강수량(mm)", "생산량(톤)"),
        ("연간평균가격(원)", "생산량(톤)"),
    ]
    colors = ["#d7191c", "#2c7bb6", "#74c476"]

    for ax, (x_col, y_col), color in zip(axes, pairs, colors):
        ax.scatter(df[x_col], df[y_col], s=80, color=color, edgecolors="black", linewidth=0.8, zorder=3)
        for year, row in df.iterrows():
            ax.annotate(str(year), (row[x_col], row[y_col]),
                        textcoords="offset points", xytext=(5, 4), fontsize=8)
        r = df[x_col].corr(df[y_col])
        ax.set_xlabel(x_col, fontsize=9)
        ax.set_ylabel(y_col, fontsize=9)
        ax.set_title(f"r = {r:.3f}", fontsize=11)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, save_dir / "12_production_vs_weather_price.png")


# ──────────────────────────────────────────────
# Main entry
# ──────────────────────────────────────────────

def run_all_eda(save_dir: str | Path = SAVE_DIR) -> None:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    weather, price, merged = load_data()
    daily = _daily_mean(weather)

    plot_weather_timeseries(daily, save_dir)
    plot_spatial_heatmap(weather, save_dir)
    plot_weather_distributions(weather, save_dir)
    plot_weather_boxplot_by_year(weather, save_dir)
    plot_seasonal_decomposition(daily, save_dir)
    plot_acf_pacf(daily, save_dir)
    plot_rice_price_timeseries(price, save_dir)
    plot_rice_price_distribution(price, save_dir)
    plot_annual_production(merged, save_dir)
    plot_weather_price_correlation(daily, price, save_dir)
    plot_monthly_weather_heatmap(daily, save_dir)
    plot_production_vs_weather_price(daily, price, merged, save_dir)

    print(f"\n[EDA] All plots saved to: {save_dir}")


if __name__ == "__main__":
    run_all_eda()
