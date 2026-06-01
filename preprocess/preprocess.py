import os
import pandas as pd
import numpy as np

import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

def _find_file(pattern: str) -> str:
    matches = glob.glob(os.path.join(DATA_DIR, pattern))
    if not matches:
        raise FileNotFoundError(f"{DATA_DIR} 안에 '{pattern}' 파일이 없습니다.")
    return sorted(matches)[-1]  # 여러 개면 가장 최신 파일 사용

# 강수량
rn_raw = open(_find_file('rn_*.csv'), encoding='cp949').read()
rn_lines = [l.strip() for l in rn_raw.split('\n') if l.strip()]
rn_data = [l for l in rn_lines if l.startswith('20')]
rn_rows = []
for l in rn_data:
    parts = l.split(',')
    rn_rows.append({'date': parts[0], 'rainfall_mm': parts[2] if parts[2] else None})
df_rn = pd.DataFrame(rn_rows)
df_rn['date'] = pd.to_datetime(df_rn['date'])
df_rn['rainfall_mm'] = pd.to_numeric(df_rn['rainfall_mm'], errors='coerce').fillna(0)

# 기온
ta_raw = open(_find_file('ta_*.csv'), encoding='cp949').read()
ta_lines = [l.strip() for l in ta_raw.split('\n') if l.strip()]
ta_data = [l for l in ta_lines if any(y in l for y in ['2020','2021','2022','2023','2024','2025','2026'])]
ta_rows = []
for l in ta_data:
    parts = l.split(',')
    if len(parts) >= 5:
        ta_rows.append({'date': parts[0].strip(), 'temp_avg': parts[2].strip(), 'temp_min': parts[3].strip(), 'temp_max': parts[4].strip()})
df_ta = pd.DataFrame(ta_rows)
df_ta['date'] = pd.to_datetime(df_ta['date'])
for col in ['temp_avg','temp_min','temp_max']:
    df_ta[col] = pd.to_numeric(df_ta[col], errors='coerce')

# 기상 merge
df_weather = pd.merge(df_rn, df_ta, on='date', how='inner')

# 생산량 (2019년 앵커 추가해서 보간 시작점 확보)
production = {
    2019: 80196,  # 2020과 동일값으로 앵커
    2020: 80196,
    2021: 110029,
    2022: 96105,
    2023: 93799,
    2024: 94778,
    2025: 93393,
}
prod_points = pd.DataFrame([
    {'date': pd.Timestamp(f'{year}-12-31'), 'production_ton': val}
    for year, val in production.items()
])

date_range = pd.date_range(df_weather['date'].min(), df_weather['date'].max(), freq='D')
df_prod = pd.DataFrame({'date': date_range})
df_prod = pd.merge(df_prod, prod_points, on='date', how='left')
df_prod['production_ton'] = df_prod['production_ton'].interpolate(method='linear').bfill()

# 최종 merge
df_final = pd.merge(df_weather, df_prod, on='date', how='left')
df_final = df_final.dropna(subset=['temp_avg'])

print(f"최종 데이터: {df_final.shape}")
print(f"NaN 개수: {df_final.isna().sum().to_dict()}")
print(df_final.head(5).to_string())
print(df_final.tail(5).to_string())

df_final.to_csv(os.path.join(DATA_DIR, 'haenam_merged.csv'), index=False, encoding='utf-8-sig')
print("\n저장완료:", os.path.join(DATA_DIR, 'haenam_merged.csv'))
