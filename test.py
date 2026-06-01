import os
import io
import requests
import xarray as xr
import pandas as pd
import numpy as np

# 글로벌 변수로 위경도 맵 캐싱
LAT_LON_MAP = None


def get_latlon_map(auth_key):
    """기상청 1.6 API를 텍스트 형태로 읽어와 nx, ny 격자에 대응하는 위경도 판을 생성합니다."""
    global LAT_LON_MAP
    if LAT_LON_MAP is not None:
        return LAT_LON_MAP

    print("\n🌐 [최초 1회 실행] 기상청 격자 위경도 매핑 판(Grid Map) 다운로드 중... (약 10~20초 소요)")

    lat_url = f"https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-sfc_obs_latlon_api?latlon=lat&authKey={auth_key}"
    lon_url = f"https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-sfc_obs_latlon_api?latlon=lon&authKey={auth_key}"

    try:
        # 1. 위도 데이터 가져오기 및 파싱
        lat_res = requests.get(lat_url)
        if "인증 오류" in lat_res.text or "error" in lat_res.text.lower():
            print("❌ 기상청 서버 메시지:", lat_res.text[:200])
            return None

        # low_memory=False 추가 및 일단 문자열로 안전하게 읽기
        df_lat = pd.read_csv(io.StringIO(lat_res.text), sep=r'\s+', comment='#', header=None, names=['ny', 'nx', 'lat'],
                             low_memory=False)

        # 2. 경도 데이터 가져오기 및 파싱
        lon_res = requests.get(lon_url)
        df_lon = pd.read_csv(io.StringIO(lon_res.text), sep=r'\s+', comment='#', header=None, names=['ny', 'nx', 'lon'],
                             low_memory=False)

        # [오류 해결 핵심] 숫자가 아닌 잘못된 행(헤더 텍스트 등) 제거 및 데이터 타입 숫자형(int/float)으로 강제 변환
        for df in [df_lat, df_lon]:
            df['ny'] = pd.to_numeric(df['ny'], errors='coerce')
            df['nx'] = pd.to_numeric(df['nx'], errors='coerce')

        # 결측치(문자열이라 변환 실패한 행) 제거 후 int 타입으로 통일
        df_lat = df_lat.dropna(subset=['ny', 'nx']).astype({'ny': 'int64', 'nx': 'int64'})
        df_lon = df_lon.dropna(subset=['ny', 'nx']).astype({'ny': 'int64', 'nx': 'int64'})

        # 위도/경도 값도 숫자로 변환
        df_lat['lat'] = pd.to_numeric(df_lat['lat'], errors='coerce')
        df_lon['lon'] = pd.to_numeric(df_lon['lon'], errors='coerce')
        df_lat = df_lat.dropna(subset=['lat'])
        df_lon = df_lon.dropna(subset=['lon'])

        # 3. ny, nx (격자좌표) 기준으로 병합
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
        ds = xr.open_dataset(temp_nc_file)
        data_var = list(ds.data_vars)[0]

        # 3. 데이터프레임 변환 후 컬럼명 소문자 규격화
        df = ds[data_var].to_dataframe().reset_index()
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={data_var.lower(): 'value'})

        # [오류 해결 핵심] 다운로드한 날씨 파일의 데이터 타입도 int64로 확실하게 일치시킴
        df = df.astype({'ny': 'int64', 'nx': 'int64'})

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

        # 7. CSV 파일로 저장
        if not os.path.exists(output_csv_path):
            final_df.to_csv(output_csv_path, index=False, mode='w', encoding='utf-8-sig')
        else:
            final_df.to_csv(output_csv_path, index=False, mode='a', header=False, encoding='utf-8-sig')

        print(f"[{tm_str} | {obs_element}] 해남군 격자 {len(final_df)}개 추출 및 CSV 저장 성공!")

    except Exception as e:
        print(f"❌ 처리 중 오류 발생: {e}")

    finally:
        if os.path.exists(temp_nc_file):
            os.remove(temp_nc_file)


# ==========================================
#                  실행부
# ==========================================
if __name__ == "__main__":

    AUTH_KEY = "5uVnBqgbS8KlZwaoG8vC8w"  # 본인의 기상청 인증키 입력
    OUTPUT_FILE = "haenam_weather_data.csv"

    # 테스트용 시간 범위
    dates = pd.date_range(start="2010-01-01 00:00", end="2010-01-01 02:00", freq="h")
    target_times = [date.strftime("%Y%m%d%H%M") for date in dates]

    # 기온(ta)과 일강수(rn_day) 지정
    elements = ['ta', 'rn_day']

    for tm in target_times:
        for obs in elements:
            download_and_extract_haenam(tm, obs, AUTH_KEY, OUTPUT_FILE)

    print("\n🎉 모든 지정된 데이터 처리가 완료되었습니다!")