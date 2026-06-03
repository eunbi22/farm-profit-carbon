import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

OUTPUT = os.path.join(os.path.dirname(__file__), "plots", "box_plot.png")
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

df = pd.read_csv("../data/haenam_weather_historical.csv", dtype=str)
df['ta'] = pd.to_numeric(df['ta'], errors='coerce')
df['rn_day'] = pd.to_numeric(df['rn_day'], errors='coerce')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("해남군 기상 데이터 박스플롯", fontsize=14, fontweight='bold')

ax1.boxplot(df['ta'].dropna(), patch_artist=True,
            boxprops=dict(facecolor='lightsalmon'))
ax1.set_title("기온 (ta)")
ax1.set_ylabel("기온 (°C)")
ax1.grid(True, alpha=0.3, axis='y')

ax2.boxplot(df['rn_day'].dropna(), patch_artist=True,
            boxprops=dict(facecolor='lightsteelblue'))
ax2.set_title("강수량 (rn_day)")
ax2.set_ylabel("강수량 (mm)")
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight')
print(f"저장 완료: {OUTPUT}")
