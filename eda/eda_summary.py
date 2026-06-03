import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("../data/haenam_weather_historical.csv", dtype=str)
df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
df['ta'] = pd.to_numeric(df['ta'], errors='coerce')
df['rn_day'] = pd.to_numeric(df['rn_day'], errors='coerce')

for col, label in [('ta', '기온 (°C)'), ('rn_day', '강수량 (mm)')]:
    s = df[col].dropna()
    print(f"\n{'='*40}")
    print(f"  {label} 기술통계")
    print(f"{'='*40}")
    print(f"  평균(Mean)      : {s.mean():.4f}")
    print(f"  중앙값(Median)  : {s.median():.4f}")
    print(f"  최빈값(Mode)    : {s.mode().iloc[0]:.4f}")
    print(f"  최솟값(Min)     : {s.min():.4f}")
    print(f"  최댓값(Max)     : {s.max():.4f}")
    print(f"  표준편차(Std)   : {s.std():.4f}")

print("\n\n[df.describe()]")
print(df[['ta', 'rn_day']].describe())
