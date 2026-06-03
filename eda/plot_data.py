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
