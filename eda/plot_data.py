import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

DATA_PATH = "../data/haenam_weather_historical.csv"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "plots")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "weather_analysis.png")

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH, dtype=str)
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df['ta'] = pd.to_numeric(df['ta'], errors='coerce')
df['rn_day'] = pd.to_numeric(df['rn_day'], errors='coerce')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
fig.suptitle("해남군 기상 데이터 (2020~2025)", fontsize=14, fontweight='bold')

ax1.plot(df['date'], df['ta'], color='red', linewidth=0.7, label='기온 (ta)')
ax1.set_ylabel("기온 (°C)")
ax1.set_title("일 평균 기온")
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

ax2.bar(df['date'], df['rn_day'], color='steelblue', width=1.0, label='강수량 (rn_day)')
ax2.set_ylabel("강수량 (mm)")
ax2.set_title("일 강수량")
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3, axis='y')

ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
print(f"저장 완료: {OUTPUT_FILE}")

# ─── 쌀 20KG 중도매인 판매가격 ────────────────────────────────────────────────
PRICE_PATH = "../data/쌀20KG_중도매인판매가격_통합.csv"
PRICE_OUTPUT = os.path.join(OUTPUT_DIR, "price_analysis.png")

price_df = pd.read_excel(PRICE_PATH)
price_df.columns = ["date", "price", "change"]
price_df["date"]   = pd.to_datetime(price_df["date"])
price_df["price"]  = pd.to_numeric(price_df["price"], errors="coerce")
price_df["change"] = pd.to_numeric(price_df["change"], errors="coerce")
price_df = price_df.dropna(subset=["price"]).sort_values("date").reset_index(drop=True)
price_df["year"]  = price_df["date"].dt.year
price_df["month"] = price_df["date"].dt.month
price_df["ma30"]  = price_df["price"].rolling(30, center=True, min_periods=1).mean()

fig2, axes = plt.subplots(3, 1, figsize=(16, 12))
fig2.suptitle("쌀 20KG 중도매인 판매가격 분석", fontsize=14, fontweight="bold")

# ① 전체 시계열 + 30일 이동평균
ax = axes[0]
ax.plot(price_df["date"], price_df["price"],
        color="steelblue", linewidth=0.8, alpha=0.6, label="일별 가격")
ax.plot(price_df["date"], price_df["ma30"],
        color="crimson", linewidth=2.0, label="30일 이동평균")
ax.set_ylabel("가격 (원/20kg)")
ax.set_title("전체 기간 가격 추이")
ax.legend(loc="upper left")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

# ② 연도별 평균가격 막대
ax = axes[1]
yearly = price_df.groupby("year")["price"].agg(["mean", "min", "max"]).reset_index()
bars = ax.bar(yearly["year"].astype(str), yearly["mean"],
              color="steelblue", edgecolor="white", zorder=3)
ax.errorbar(range(len(yearly)),
            yearly["mean"],
            yerr=[yearly["mean"] - yearly["min"], yearly["max"] - yearly["mean"]],
            fmt="none", color="gray", capsize=5, linewidth=1.2, zorder=4)
for bar, val in zip(bars, yearly["mean"]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200, f"{int(val):,}원",
            ha="center", va="bottom", fontsize=9)
ax.set_ylabel("평균 가격 (원/20kg)")
ax.set_title("연도별 평균 가격 (오차막대: 연간 최소~최대)")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(True, axis="y", alpha=0.3, zorder=0)

# ③ 월별 계절성 (연도별 컬러 라인)
ax = axes[2]
years = sorted(price_df["year"].unique())
cmap  = plt.cm.get_cmap("tab10", len(years))
for i, yr in enumerate(years):
    monthly_avg = (
        price_df[price_df["year"] == yr]
        .groupby("month")["price"].mean()
        .reindex(range(1, 13))
    )
    ax.plot(monthly_avg.index, monthly_avg.values,
            marker="o", markersize=4, linewidth=1.6,
            color=cmap(i), label=str(yr))
ax.set_xlabel("월")
ax.set_ylabel("월평균 가격 (원/20kg)")
ax.set_title("월별 계절성 패턴 (연도별)")
ax.set_xticks(range(1, 13))
ax.set_xticklabels([f"{m}월" for m in range(1, 13)])
ax.legend(title="연도", bbox_to_anchor=(1.01, 1), loc="upper left")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PRICE_OUTPUT, dpi=150, bbox_inches="tight")
print(f"저장 완료: {PRICE_OUTPUT}")
