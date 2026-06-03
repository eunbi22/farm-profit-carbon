import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

OUTPUT = os.path.join(os.path.dirname(__file__), "plots", "bar_histogram.png")
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

df = pd.read_csv("../data/haenam_weather_historical.csv", dtype=str)
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df['ta'] = pd.to_numeric(df['ta'], errors='coerce')

yearly_mean = df.groupby(df['date'].dt.year)['ta'].mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("해남군 기온 분포 분석", fontsize=14, fontweight='bold')

ax1.hist(df['ta'].dropna(), bins=40, color='tomato', edgecolor='white', linewidth=0.5)
ax1.set_title("기온 히스토그램")
ax1.set_xlabel("기온 (°C)")
ax1.set_ylabel("빈도")
ax1.grid(True, alpha=0.3, axis='y')

ax2.bar(yearly_mean.index, yearly_mean.values, color='steelblue', edgecolor='white')
ax2.set_title("연도별 평균 기온")
ax2.set_xlabel("연도")
ax2.set_ylabel("평균 기온 (°C)")
ax2.set_xticks(yearly_mean.index)
for x, y in zip(yearly_mean.index, yearly_mean.values):
    ax2.text(x, y + 0.1, f"{y:.1f}", ha='center', va='bottom', fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight')
print(f"저장 완료: {OUTPUT}")
