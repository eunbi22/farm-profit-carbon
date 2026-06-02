import os
import io
import requests
import xarray as xr
import pandas as pd
import numpy as np

# 글로벌 변수로 위경도 맵 캐싱
LAT_LON_MAP = None


def _parse_kma_latlon_text(text, value_col):
    """
    기상청 1.6 API 텍스트 응답 파싱.
    형식: 첫 줄 "  nx,  ny,=" → 이후 줄들은 쉼표 구분 float 값 (인덱스 없음)
    """
    first_nl = text.index('\n')
    header_parts = text[:first_nl].replace('=', '').split(',')
    nx = int(header_parts[0].strip())
    ny = int(header_parts[1].strip())

    data_io = io.StringIO(text[first_nl + 1:])
    arr = pd.read_csv(data_io, header=None, dtype=np.float64).values.flatten()
    vals = arr[~np.isnan(arr)]

    if len(vals) != nx * ny:
        raise ValueError(f"예상 값 수 {nx * ny}개, 실제 {len(vals)}개")

    vals = vals.reshape(ny, nx)
    ny_idx, nx_idx = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
    return pd.DataFrame({
        'ny': ny_idx.flatten().astype(np.int64),
        'nx': nx_idx.flatten().astype(np.int64),
        value_col: vals.flatten(),
    })


def get_latlon_map(auth_key):
    """기상청 1.6 API로 nx, ny 격자에 대응하는 위경도 판을 생성합니다."""
    global LAT_LON_MAP
    if LAT_LON_MAP is not None:
        return LAT_LON_MAP

    print("\n🌐 [최초 1회 실행] 기상청 격자 위경도 매핑 판(Grid Map) 다운로드 중... (약 10~20초 소요)")

    lat_url = f"https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-sfc_obs_latlon_api?latlon=lat&authKey={auth_key}"
    lon_url = f"https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-sfc_obs_latlon_api?latlon=lon&authKey={auth_key}"

    try:
        lat_res = requests.get(lat_url, timeout=60)
        lon_res = requests.get(lon_url, timeout=60)

        if lat_res.status_code != 200:
            print(f"❌ 위도 API HTTP 오류: {lat_res.status_code}")
            return None
        if "인증 오류" in lat_res.text[:300]:
            print("❌ 인증 오류:", lat_res.text[:200])
            return None

        df_lat = _parse_kma_latlon_text(lat_res.text, 'lat')
        df_lon = _parse_kma_latlon_text(lon_res.text, 'lon')

        LAT_LON_MAP = pd.merge(df_lat, df_lon, on=['ny', 'nx'])
        print(f"🌐 위경도 매핑 판 구축 완료! (총 {len(LAT_LON_MAP)}개 격자 매핑됨)\n")

    except Exception as e:
        print(f"❌ 위경도 매핑 판 구축 실패: {e}")
        LAT_LON_MAP = None

    return LAT_LON_MAP


def download_and_extract_haenam(tm_str, obs_element, auth_key, output_csv_path):
    # 위경도 지도 가져오기
    grid_map = get_latlon_map(auth_key)
    if grid_map is None or grid_map.empty:
        print("❌ 위경도 매핑 데이터가 유효하지 않아 수집을 중단합니다.")
        return

    # 1. 기상청 1.4 파일 다운로드 API URL 설정
    url = f"https://apihub.kma.go.kr/api/typ01/url/sfc_grid_nc_down.php?obs={obs_element}&tm={tm_str}&authKey={auth_key}"
    temp_nc_file = f"temp_{obs_element}_{tm_str}.nc"

    print(f"[{tm_str} | {obs_element}] 기상청에서 대한민국 전체 NetCDF 파일 다운로드 중...")
    response = requests.get(url)

    if response.status_code != 200:
        print(f"API 요청 실패! HTTP 상태 코드: {response.status_code}")
        return

    with open(temp_nc_file, "wb") as f:
        f.write(response.content)

    try:
        # 2. xarray로 알맹이 NetCDF 파일 열기
        with xr.open_dataset(temp_nc_file) as ds:
            data_var = list(ds.data_vars)[0]

            # 3. 데이터프레임 변환 후 컬럼명 소문자 규격화
            df = ds[data_var].to_dataframe().reset_index()

        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={data_var.lower(): 'value'})

        # [오류 해결 핵심] 다운로드한 날씨 파일의 데이터 타입도 int64로 확실하게 일치시킴
        df = df.astype({'ny': 'int64', 'nx': 'int64'})

        # 진단: ny/nx 범위 확인 (grid_map과 인덱스 범위가 맞지 않으면 병합 결과가 0이 됨)
        print(f"[DEBUG] NetCDF ny: {df['ny'].min()}~{df['ny'].max()}, nx: {df['nx'].min()}~{df['nx'].max()}")
        print(f"[DEBUG] grid_map ny: {grid_map['ny'].min()}~{grid_map['ny'].max()}, nx: {grid_map['nx'].min()}~{grid_map['nx'].max()}")

        # 4. 다운로드한 데이터에 위경도 매핑 지도 융합
        df = pd.merge(df, grid_map, on=['ny', 'nx'], how='inner')

        # 5. 해남군 위경도 범위로 필터링
        lat_min, lat_max = 34.28, 34.67
        lon_min, lon_max = 126.24, 126.75

        haenam_df = df[
            (df['lat'] >= lat_min) & (df['lat'] <= lat_max) &
            (df['lon'] >= lon_min) & (df['lon'] <= lon_max)
            ].copy()

        if haenam_df.empty:
            print(f"⚠️ 해남군 위경도 범위 내에 매칭되는 격자가 없습니다.")
            return

        # 6. 저장용 포맷 정리
        haenam_df['time_kst'] = tm_str
        haenam_df['element'] = obs_element

        final_df = haenam_df[['time_kst', 'element', 'lat', 'lon', 'value']]

        # 7. CSV 파일로 저장 (output_csv_path가 None이면 저장 없이 반환만)
        if output_csv_path is not None:
            if not os.path.exists(output_csv_path):
                final_df.to_csv(output_csv_path, index=False, mode='w', encoding='utf-8-sig')
            else:
                final_df.to_csv(output_csv_path, index=False, mode='a', header=False, encoding='utf-8-sig')
            print(f"[{tm_str} | {obs_element}] 해남군 격자 {len(final_df)}개 추출 및 CSV 저장 성공!")

        return final_df

    except Exception as e:
        print(f"❌ 처리 중 오류 발생: {e}")
        return None

    finally:
        if os.path.exists(temp_nc_file):
            os.remove(temp_nc_file)


# ==========================================
#                  실행부
# ==========================================
if __name__ == "__main__":

    AUTH_KEY = "5uVnBqgbS8KlZwaoG8vC8w"
    OUTPUT_FILE = "haenam_weather_data.csv"

    # 누적/일값 요소: 하루 중 1회(00시)만 다운로드
    DAILY_ELEMENTS = {'rn_day', 'sd_tot', 'sd_day', 'sd_24h'}

    # 시간형 요소: 00, 06, 12, 18시 4회 다운로드 후 일평균 산출
    HOURLY_ELEMENTS = ['ta', 'hm', 'td', 'ws_10m', 'pa', 'ps', 'vs', 'ta_chi']

    dates = pd.date_range(start="2010-01-01", end="2010-01-03", freq="D")

    for date in dates:
        date_str = date.strftime("%Y%m%d")

        # 누적형: 00시 1회
        for obs in DAILY_ELEMENTS:
            tm = date.strftime("%Y%m%d0000")
            download_and_extract_haenam(tm, obs, AUTH_KEY, OUTPUT_FILE)

        # 시간형: 4회 다운로드 후 일평균
        for obs in HOURLY_ELEMENTS:
            frames = []
            for hour in ["0000", "0600", "1200", "1800"]:
                tm = date.strftime("%Y%m%d") + hour
                df = download_and_extract_haenam(tm, obs, AUTH_KEY, output_csv_path=None)
                if df is not None:
                    frames.append(df)

            if not frames:
                continue

            daily_mean = (
                pd.concat(frames)
                .groupby(['lat', 'lon'], as_index=False)['value']
                .mean()
            )
            daily_mean['time_kst'] = date_str
            daily_mean['element'] = obs
            final = daily_mean[['time_kst', 'element', 'lat', 'lon', 'value']]

            if not os.path.exists(OUTPUT_FILE):
                final.to_csv(OUTPUT_FILE, index=False, mode='w', encoding='utf-8-sig')
            else:
                final.to_csv(OUTPUT_FILE, index=False, mode='a', header=False, encoding='utf-8-sig')

            print(f"[{date_str} | {obs}] 일평균 저장 완료 ({len(final)}개 격자)")

    print("\n모든 지정된 데이터 처리가 완료되었습니다!")