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

# 생산량: 연도별 확정값을 해당 연도 전체 날짜에 매핑
production = {
    2020: 80196,
    2021: 110029,
    2022: 96105,
    2023: 93799,
    2024: 94778,
    2025: 93393,
}
last_prod_year = max(production.keys())

df_weather['year'] = df_weather['date'].dt.year
df_weather = df_weather[df_weather['year'] <= last_prod_year].drop(columns='year')

df_weather['production_ton'] = df_weather['date'].dt.year.map(production)

# 최종 merge
df_final = df_weather.dropna(subset=['temp_avg', 'production_ton'])

print(f"최종 데이터: {df_final.shape}")
print(f"NaN 개수: {df_final.isna().sum().to_dict()}")
print(df_final.head(5).to_string())
print(df_final.tail(5).to_string())

df_final.to_csv(os.path.join(DATA_DIR, 'haenam_merged.csv'), index=False, encoding='utf-8-sig')
print("\n저장완료:", os.path.join(DATA_DIR, 'haenam_merged.csv'))
