import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

OUTPUT = os.path.join(os.path.dirname(__file__), "plots", "correlation_heatmap.png")
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

df = pd.read_csv("../data/haenam_weather_historical.csv", dtype=str)
df['ta'] = pd.to_numeric(df['ta'], errors='coerce')
df['rn_day'] = pd.to_numeric(df['rn_day'], errors='coerce')

corr = df[['ta', 'rn_day']].corr(method='pearson')
print("Pearson 상관계수 행렬:")
print(corr)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    corr,
    annot=True,
    fmt=".4f",
    cmap='coolwarm',
    vmin=-1, vmax=1,
    linewidths=0.5,
    ax=ax,
    xticklabels=['기온 (ta)', '강수량 (rn_day)'],
    yticklabels=['기온 (ta)', '강수량 (rn_day)'],
)
ax.set_title("기온-강수량 피어슨 상관계수 히트맵", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight')
print(f"저장 완료: {OUTPUT}")
