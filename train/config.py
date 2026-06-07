import os

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT, "data")
RESULT_DIR = os.path.join(ROOT, "train", "result")

FARMMAP_CSV    = os.path.join(DATA_DIR, "haenam_farmmap_20260527_1312.csv")
FARMMAP_GEO    = os.path.join(DATA_DIR, "haenam_farmmap_20260527_1312.geojson")
WEATHER_CSV    = os.path.join(DATA_DIR, "haenam_weather_grid.csv")
PRODUCTION_CSV = os.path.join(DATA_DIR, "시군별_논벼_생산량_정곡_92.9__20260601192918.csv")

GROWING_MONTHS   = [5, 6, 7, 8, 9]   # May–Sep (벼 영농기)
TA_SCALE         = 0.1                # KMA raw ta 단위: 0.1°C
RN_SCALE         = 1.0                # rn_day 단위: mm

DATA_START_DATE  = "2020-05-02"
TRAIN_YEARS_END  = 2023   # 학습 마지막 연도 (포함)
TEST_YEARS_START = 2024   # 테스트 첫 연도

CV_FOLDS      = 2
OPTUNA_TRIALS = 50
RANDOM_SEED   = 42

for _d in [
    RESULT_DIR,
    os.path.join(RESULT_DIR, "models"),
    os.path.join(RESULT_DIR, "metrics"),
]:
    os.makedirs(_d, exist_ok=True)
