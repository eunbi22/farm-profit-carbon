"""
해남군 13개 읍면(面) 단위 기상청 격자 데이터 수집 코드
=======================================================
[데이터 흐름]
  기상청 API (NetCDF) → 격자 위경도 매핑 → Shapefile 공간 조인
  → 읍면별 일평균 집계 → CSV 저장

[사전 준비]
  pip install geopandas shapely xarray requests pandas numpy fiona pyproj

[Shapefile 준비]
  통계청 SGIS (https://sgis.kostat.go.kr) 또는
  국가공간정보포털 (https://www.nsdi.go.kr) 에서
  "읍면동 경계" SHP 파일 다운로드 후 아래 경로에 배치:
    ./shp/LARD_ADM_SECT_SGG_전라남도.shp  (시군구 단위)
  또는
    ./shp/LARD_ADM_SECT_UMD_전라남도.shp  (읍면동 단위)  ← 권장

  ※ Shapefile이 없을 경우 → 자동으로 '중심좌표 근사' 모드로 전환
"""

import os
import io
import warnings
import requests
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

# ============================================================
# 0. 설정값
# ============================================================
AUTH_KEY   = "YOUR_KMA_API_KEY"        # ← 기상청 API 키 입력
OUTPUT_DIR = Path("./output")
SHP_PATH   = Path("./shp/읍면동_경계.shp")  # ← Shapefile 경로 (없으면 근사 모드)

OUTPUT_DIR.mkdir(exist_ok=True)

# 해남군 13개 읍면 행정코드 (법정동 코드 앞 10자리)
HAENAM_EMD_CODES = {
    '해남읍': '4681025000',
    '삼산면': '4681031000',
    '화산면': '4681032000',
    '현산면': '4681033000',
    '송지면': '4681034000',
    '북평면': '4681035000',
    '북일면': '4681036000',
    '옥천면': '4681037000',
    '계곡면': '4681038000',
    '마산면': '4681039000',
    '황산면': '4681040000',
    '산이면': '4681041000',
    '문내면': '4681042000',
}

# 방법 2용: 읍면 중심 근사 좌표 (Shapefile 없을 때 사용)
HAENAM_MYEON_CENTERS = {
    '해남읍': (34.574, 126.599),
    '삼산면': (34.627, 126.522),
    '화산면': (34.556, 126.522),
    '현산면': (34.504, 126.561),
    '송지면': (34.382, 126.591),
    '북평면': (34.469, 126.704),
    '북일면': (34.518, 126.671),
    '옥천면': (34.568, 126.649),
    '계곡면': (34.618, 126.648),
    '마산면': (34.619, 126.698),
    '황산면': (34.527, 126.730),
    '산이면': (34.464, 126.637),
    '문내면': (34.417, 126.484),
}

# 해남군 전체 바운딩 박스 (사전 필터링용 - 속도 향상)
HAENAM_BBOX = {
    'lat_min': 34.28, 'lat_max': 34.67,
    'lon_min': 126.24, 'lon_max': 126.75,
}

# 요소 분류
DAILY_ELEMENTS  = {'rn_day', 'sd_tot', 'sd_day', 'sd_24h'}   # 00시 1회
HOURLY_ELEMENTS = ['ta', 'hm', 'td', 'ws_10m', 'pa', 'ps', 'vs', 'ta_chi']

# 글로벌 캐시
LAT_LON_MAP   = None
HAENAM_GDF    = None   # Shapefile 기반 해남군 읍면 GeoDataFrame
USE_SHAPEFILE = None   # True: Shapefile / False: 근사 모드


# ============================================================
# 1. 위경도 격자 매핑 (기상청 1.6 API)
# ============================================================
def _parse_kma_latlon_text(text, value_col):
    first_nl = text.index('\n')
    header_parts = text[:first_nl].replace('=', '').split(',')
    nx = int(header_parts[0].strip())
    ny = int(header_parts[1].strip())

    data_io = io.StringIO(text[first_nl + 1:])
    arr  = pd.read_csv(data_io, header=None, dtype=np.float64).values.flatten()
    vals = arr[~np.isnan(arr)]

    if len(vals) != nx * ny:
        raise ValueError(f"예상 {nx * ny}개, 실제 {len(vals)}개")

    vals = vals.reshape(ny, nx)
    ny_idx, nx_idx = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    return pd.DataFrame({
        'ny': ny_idx.flatten().astype(np.int64),
        'nx': nx_idx.flatten().astype(np.int64),
        value_col: vals.flatten(),
    })


def get_latlon_map(auth_key):
    global LAT_LON_MAP
    if LAT_LON_MAP is not None:
        return LAT_LON_MAP

    print("🌐 기상청 격자 위경도 매핑 다운로드 중... (최초 1회, 약 10~20초)")
    base = "https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-sfc_obs_latlon_api"
    try:
        lat_res = requests.get(f"{base}?latlon=lat&authKey={auth_key}", timeout=60)
        lon_res = requests.get(f"{base}?latlon=lon&authKey={auth_key}", timeout=60)

        if "인증 오류" in lat_res.text[:300]:
            print("❌ API 인증 오류:", lat_res.text[:200])
            return None

        df_lat = _parse_kma_latlon_text(lat_res.text, 'lat')
        df_lon = _parse_kma_latlon_text(lon_res.text, 'lon')
        LAT_LON_MAP = pd.merge(df_lat, df_lon, on=['ny', 'nx'])
        print(f"✅ 위경도 매핑 완료 ({len(LAT_LON_MAP):,}개 격자)\n")
    except Exception as e:
        print(f"❌ 위경도 매핑 실패: {e}")
        LAT_LON_MAP = None
    return LAT_LON_MAP


# ============================================================
# 2-A. 읍면 할당 방법: Shapefile 기반 (정밀)
# ============================================================
def load_haenam_shapefile():
    """
    읍면동 Shapefile을 로드해 해남군 13개 읍면만 추출.
    반환: GeoDataFrame (컬럼: EMD_NM, geometry) or None
    """
    global HAENAM_GDF
    if HAENAM_GDF is not None:
        return HAENAM_GDF

    if not SHP_PATH.exists():
        return None

    try:
        import geopandas as gpd
        print(f"📂 Shapefile 로드 중: {SHP_PATH}")
        gdf = gpd.read_file(SHP_PATH, encoding='cp949')
        gdf = gdf.to_crs(epsg=4326)  # WGS84 변환

        # 컬럼명 탐색 (기관마다 다름)
        possible_sgg = ['SGG_NM', 'SIG_KOR_NM', 'SIGUNGU_NM', 'sig_kor_nm']
        possible_emd = ['EMD_NM', 'ADM_NM', 'UMD_NM', 'adm_nm']

        sgg_col = next((c for c in possible_sgg if c in gdf.columns), None)
        emd_col = next((c for c in possible_emd if c in gdf.columns), None)

        if sgg_col is None or emd_col is None:
            print(f"⚠️  Shapefile 컬럼 탐색 실패. 실제 컬럼: {list(gdf.columns)}")
            return None

        # 해남군 + 13개 읍면만 필터
        target_names = set(HAENAM_EMD_CODES.keys())
        HAENAM_GDF = gdf[
            (gdf[sgg_col] == '해남군') &
            (gdf[emd_col].isin(target_names))
        ][['geometry', emd_col]].rename(columns={emd_col: 'myeon'}).reset_index(drop=True)

        if HAENAM_GDF.empty:
            print("⚠️  Shapefile에서 해남군 읍면을 찾지 못했습니다.")
            HAENAM_GDF = None
            return None

        print(f"✅ Shapefile 로드 완료 ({len(HAENAM_GDF)}개 읍면 폴리곤)\n")
        return HAENAM_GDF

    except ImportError:
        print("⚠️  geopandas 미설치 → 근사 모드로 전환 (pip install geopandas)")
        return None
    except Exception as e:
        print(f"⚠️  Shapefile 로드 오류: {e} → 근사 모드로 전환")
        return None


def assign_myeon_shapefile(df_haenam):
    """
    격자 DataFrame에 Shapefile 폴리곤 기반으로 myeon 컬럼 추가.
    df_haenam: lat, lon 컬럼 포함 DataFrame
    """
    import geopandas as gpd
    from shapely.geometry import Point

    gdf_grid = gpd.GeoDataFrame(
        df_haenam,
        geometry=gpd.points_from_xy(df_haenam['lon'], df_haenam['lat']),
        crs='EPSG:4326'
    )

    # 공간 조인 (격자점 → 읍면 폴리곤)
    joined = gpd.sjoin(gdf_grid, HAENAM_GDF, how='left', predicate='within')
    df_haenam = df_haenam.copy()
    df_haenam['myeon'] = joined['myeon'].values

    matched = df_haenam['myeon'].notna().sum()
    print(f"   📍 폴리곤 매칭: {matched}/{len(df_haenam)}개 격자 → 읍면 할당")
    return df_haenam[df_haenam['myeon'].notna()]


# ============================================================
# 2-B. 읍면 할당 방법: 중심 좌표 근사 (Shapefile 없을 때)
# ============================================================
def assign_myeon_by_distance(df_haenam):
    """
    각 격자를 가장 가까운 읍면 중심에 할당.
    df_haenam: lat, lon 컬럼 포함 DataFrame
    """
    centers = np.array(list(HAENAM_MYEON_CENTERS.values()))   # (13, 2)
    names   = list(HAENAM_MYEON_CENTERS.keys())

    lats = df_haenam['lat'].values[:, np.newaxis]   # (N, 1)
    lons = df_haenam['lon'].values[:, np.newaxis]

    # 유클리드 거리 (근사, 소규모 지역)
    dist = (lats - centers[:, 0])**2 + (lons - centers[:, 1])**2  # (N, 13)
    idx  = dist.argmin(axis=1)

    df_haenam = df_haenam.copy()
    df_haenam['myeon'] = [names[i] for i in idx]
    print(f"   📍 근사 매칭: {len(df_haenam)}개 격자 → 읍면 할당 (중심 거리 기반)")
    return df_haenam


# ============================================================
# 3. 읍면 할당 통합 함수 (자동 모드 선택)
# ============================================================
def assign_myeon(df_haenam):
    global USE_SHAPEFILE

    if USE_SHAPEFILE is None:
        gdf = load_haenam_shapefile()
        USE_SHAPEFILE = (gdf is not None)
        mode = "Shapefile(정밀)" if USE_SHAPEFILE else "중심좌표 근사"
        print(f"🗺️  읍면 할당 모드: {mode}\n")

    if USE_SHAPEFILE:
        return assign_myeon_shapefile(df_haenam)
    else:
        return assign_myeon_by_distance(df_haenam)


# ============================================================
# 4. 기상청 NetCDF 다운로드 & 해남군 추출
# ============================================================
def download_and_extract(tm_str, obs_element, auth_key, output_csv_path=None):
    """
    단일 시각·요소의 NetCDF를 다운로드해 해남군 격자 DataFrame 반환.
    output_csv_path가 None이면 저장 없이 반환만.
    """
    grid_map = get_latlon_map(auth_key)
    if grid_map is None:
        return None

    url = (
        f"https://apihub.kma.go.kr/api/typ01/url/sfc_grid_nc_down.php"
        f"?obs={obs_element}&tm={tm_str}&authKey={auth_key}"
    )
    temp_file = f"_temp_{obs_element}_{tm_str}.nc"

    try:
        res = requests.get(url, timeout=60)
        if res.status_code != 200:
            print(f"  ❌ HTTP {res.status_code} [{tm_str}|{obs_element}]")
            return None

        with open(temp_file, 'wb') as f:
            f.write(res.content)

        with xr.open_dataset(temp_file) as ds:
            var = list(ds.data_vars)[0]
            df  = ds[var].to_dataframe().reset_index()

        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={var.lower(): 'value'})
        df = df.astype({'ny': 'int64', 'nx': 'int64'})

        # 위경도 병합
        df = pd.merge(df, grid_map, on=['ny', 'nx'], how='inner')

        # 해남군 바운딩 박스 1차 필터 (속도 향상)
        df = df[
            (df['lat'] >= HAENAM_BBOX['lat_min']) & (df['lat'] <= HAENAM_BBOX['lat_max']) &
            (df['lon'] >= HAENAM_BBOX['lon_min']) & (df['lon'] <= HAENAM_BBOX['lon_max'])
        ]

        if df.empty:
            print(f"  ⚠️  [{tm_str}|{obs_element}] 해남군 범위 격자 없음")
            return None

        # 읍면 할당 (Shapefile or 근사)
        df = assign_myeon(df)

        df['time_kst']  = tm_str
        df['element']   = obs_element
        result = df[['time_kst', 'element', 'myeon', 'lat', 'lon', 'value']]

        # CSV 저장
        if output_csv_path is not None:
            header = not os.path.exists(output_csv_path)
            result.to_csv(
                output_csv_path, index=False,
                mode='w' if header else 'a',
                header=header,
                encoding='utf-8-sig'
            )
            print(f"  ✅ [{tm_str}|{obs_element}] {len(result)}개 격자 저장")

        return result

    except Exception as e:
        print(f"  ❌ [{tm_str}|{obs_element}] 오류: {e}")
        return None
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


# ============================================================
# 5. 일별 집계 저장 (읍면 × 요소별 일평균)
# ============================================================
def save_daily_mean(frames, date_str, obs_element, output_csv_path):
    """4회 시각 데이터를 읍면별 일평균으로 집계해 CSV 저장."""
    if not frames:
        print(f"  ⚠️  [{date_str}|{obs_element}] 수집 데이터 없음 → 건너뜀")
        return

    combined = pd.concat(frames)

    # 읍면별 일평균 (격자 수가 다를 수 있으니 전체 격자 평균 후 읍면 평균)
    daily = (
        combined
        .groupby(['myeon', 'lat', 'lon'], as_index=False)['value']
        .mean()
        .groupby('myeon', as_index=False)['value']
        .mean()
    )
    daily['date']    = date_str
    daily['element'] = obs_element
    daily = daily[['date', 'element', 'myeon', 'value']]

    header = not os.path.exists(output_csv_path)
    daily.to_csv(
        output_csv_path, index=False,
        mode='w' if header else 'a',
        header=header,
        encoding='utf-8-sig'
    )
    print(f"  ✅ [{date_str}|{obs_element}] 읍면별 일평균 저장 ({len(daily)}개 읍면)")


# ============================================================
# 6. 메인 실행부
# ============================================================
if __name__ == "__main__":

    # ── 수집 기간 설정 ──────────────────────────────────────
    START_DATE = "2010-01-01"
    END_DATE   = "2010-01-03"
    # ────────────────────────────────────────────────────────

    # 출력 파일 경로
    RAW_CSV   = OUTPUT_DIR / "haenam_myeon_raw.csv"      # 격자별 원시 데이터
    DAILY_CSV = OUTPUT_DIR / "haenam_myeon_daily.csv"    # 읍면별 일평균

    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
    print(f"\n{'='*60}")
    print(f"  해남군 13개 읍면 기상 데이터 수집")
    print(f"  기간: {START_DATE} ~ {END_DATE}")
    print(f"  누적 요소: {DAILY_ELEMENTS}")
    print(f"  시간 요소: {HOURLY_ELEMENTS}")
    print(f"{'='*60}\n")

    for date in dates:
        date_str = date.strftime("%Y%m%d")
        print(f"\n📅 {date_str} 처리 중...")

        # ── 누적형 요소: 00시 1회 수집, 그대로 읍면별 저장 ──
        for obs in DAILY_ELEMENTS:
            tm = date_str + "0000"
            df = download_and_extract(tm, obs, AUTH_KEY, output_csv_path=str(RAW_CSV))

            if df is not None:
                # 읍면별 집계
                myeon_daily = df.groupby('myeon', as_index=False)['value'].mean()
                myeon_daily['date']    = date_str
                myeon_daily['element'] = obs
                myeon_daily = myeon_daily[['date', 'element', 'myeon', 'value']]

                header = not DAILY_CSV.exists()
                myeon_daily.to_csv(
                    DAILY_CSV, index=False,
                    mode='w' if header else 'a',
                    header=header,
                    encoding='utf-8-sig'
                )

        # ── 시간형 요소: 00 / 06 / 12 / 18시 평균 ──────────
        for obs in HOURLY_ELEMENTS:
            frames = []
            for hour in ["0000", "0600", "1200", "1800"]:
                tm  = date_str + hour
                df  = download_and_extract(tm, obs, AUTH_KEY, output_csv_path=str(RAW_CSV))
                if df is not None:
                    frames.append(df)

            save_daily_mean(frames, date_str, obs, str(DAILY_CSV))

    print(f"\n{'='*60}")
    print(f"  ✅ 수집 완료!")
    print(f"  원시 데이터 (격자별): {RAW_CSV}")
    print(f"  일별 집계  (읍면별): {DAILY_CSV}")
    print(f"{'='*60}\n")