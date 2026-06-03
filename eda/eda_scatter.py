import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

OUTPUT = os.path.join(os.path.dirname(__file__), "plots", "scatter_plot.png")
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

df = pd.read_csv("../data/haenam_weather_historical.csv", dtype=str)
df['ta'] = pd.to_numeric(df['ta'], errors='coerce')
df['rn_day'] = pd.to_numeric(df['rn_day'], errors='coerce')
df = df.dropna(subset=['ta', 'rn_day'])

fig, ax = plt.subplots(figsize=(10, 7))

ax.scatter(df['ta'], df['rn_day'], alpha=0.3, s=10, color='steelblue', label='관측값')

coeffs = np.polyfit(df['ta'], df['rn_day'], 1)
x_line = np.linspace(df['ta'].min(), df['ta'].max(), 200)
ax.plot(x_line, np.polyval(coeffs, x_line), color='red', linewidth=2,
        label=f"회귀선: y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}")

ax.set_title("기온 vs 강수량 산점도", fontsize=13, fontweight='bold')
ax.set_xlabel("기온 (°C)")
ax.set_ylabel("강수량 (mm)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight')
print(f"저장 완료: {OUTPUT}")
