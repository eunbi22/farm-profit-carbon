import os
import io
import time
import requests
import urllib3
import xarray as xr
import pandas as pd
import numpy as np

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

AUTH_KEY = "vpPy8-qxSKCT8vPqsaigPg"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "haenam_weather_grid.csv")

LAT_MIN, LAT_MAX = 34.28, 34.67
LON_MIN, LON_MAX = 126.24, 126.75

TA_HOURS = ["0000", "0400", "0800", "1200", "1600", "2000"]

LAT_LON_MAP = None


def _parse_kma_latlon_text(text, value_col):
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
    global LAT_LON_MAP
    if LAT_LON_MAP is not None:
        return LAT_LON_MAP

    print("기상청 격자 위경도 매핑 판 다운로드 중... (최초 1회, 약 10~20초 소요)")

    lat_url = f"https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-sfc_obs_latlon_api?latlon=lat&authKey={auth_key}"
    lon_url = f"https://apihub.kma.go.kr/api/typ01/cgi-bin/url/nph-sfc_obs_latlon_api?latlon=lon&authKey={auth_key}"

    for attempt in range(1, 4):
        try:
            lat_res = requests.get(lat_url, verify=False, timeout=90)
            lon_res = requests.get(lon_url, verify=False, timeout=90)
            break
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"  위경도 맵 다운로드 실패 (시도 {attempt}/3): {e}")
            if attempt == 3:
                raise
            time.sleep(5 * attempt)

    if lat_res.status_code != 200 or "인증 오류" in lat_res.text[:300]:
        raise RuntimeError(f"위경도 API 오류: {lat_res.text[:200]}")

    df_lat = _parse_kma_latlon_text(lat_res.text, 'lat')
    df_lon = _parse_kma_latlon_text(lon_res.text, 'lon')

    LAT_LON_MAP = pd.merge(df_lat, df_lon, on=['ny', 'nx'])
    print(f"위경도 매핑 완료 ({len(LAT_LON_MAP)}개 격자)\n")
    return LAT_LON_MAP


def fetch_haenam_grid_data(tm_str, obs_element, auth_key, retries=3):
    """특정 시각의 요소 데이터를 받아 해남군 격자별 DataFrame 반환. 실패 시 None.
    반환 컬럼: ny, nx, lat, lon, value
    """
    url = (
        f"https://apihub.kma.go.kr/api/typ01/url/sfc_grid_nc_down.php"
        f"?obs={obs_element}&tm={tm_str}&authKey={auth_key}"
    )
    temp_nc = f"_tmp_{obs_element}_{tm_str}.nc"

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, verify=False, timeout=60)
            if resp.status_code != 200:
                print(f"  HTTP {resp.status_code} [{tm_str}|{obs_element}] (시도 {attempt}/{retries})")
                time.sleep(2 * attempt)
                continue

            with open(temp_nc, "wb") as f:
                f.write(resp.content)

            with xr.open_dataset(temp_nc) as ds:
                ds.load()
                data_var = list(ds.data_vars)[0]
                df = ds[data_var].to_dataframe().reset_index()

            df.columns = [c.lower() for c in df.columns]
            df = df.rename(columns={data_var.lower(): 'value'})
            df = df.astype({'ny': 'int64', 'nx': 'int64'})

            grid_map = get_latlon_map(auth_key)
            df = pd.merge(df, grid_map, on=['ny', 'nx'], how='inner')

            haenam = df[
                (df['lat'] >= LAT_MIN) & (df['lat'] <= LAT_MAX) &
                (df['lon'] >= LON_MIN) & (df['lon'] <= LON_MAX)
            ].copy()

            if haenam.empty:
                print(f"  해남 격자 없음 [{tm_str}|{obs_element}]")
                return None

            return haenam[['ny', 'nx', 'lat', 'lon', 'value']].reset_index(drop=True)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"  오류 [{tm_str}|{obs_element}] (시도 {attempt}/{retries}): {e}")
            time.sleep(10 * attempt)

        finally:
            if os.path.exists(temp_nc):
                os.remove(temp_nc)

    return None


if __name__ == "__main__":
    dates = pd.date_range(start="2023-01-01", end="2025-12-31", freq="D")

    # 이미 처리된 날짜 확인 (date 컬럼만 읽어 메모리 절약)
    done_dates = set()
    if os.path.exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE, dtype=str, usecols=['date'])
        done_dates = set(existing['date'].dropna().unique().tolist())
        print(f"기존 파일 발견: {len(done_dates)}일 이미 처리됨\n")
    else:
        pd.DataFrame(columns=['date', 'ny', 'nx', 'lat', 'lon', 'ta', 'rn_day']).to_csv(
            OUTPUT_FILE, index=False, encoding='utf-8-sig'
        )

    get_latlon_map(AUTH_KEY)

    total = len(dates)
    consecutive_failures = 0
    FAILURE_PAUSE_THRESHOLD = 5
    FAILURE_PAUSE_SECONDS = 300

    try:
        for i, date in enumerate(dates, 1):
            date_str = date.strftime("%Y%m%d")

            if date_str in done_dates:
                continue

            print(f"[{i}/{total}] {date_str} 처리 중...", end=" ", flush=True)

            # ta: 시간대별 격자 데이터 수집 후 격자(ny,nx)별 평균
            ta_frames = []
            for hour in TA_HOURS:
                df_hour = fetch_haenam_grid_data(date_str + hour, 'ta', AUTH_KEY)
                if df_hour is not None:
                    ta_frames.append(df_hour)
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= FAILURE_PAUSE_THRESHOLD:
                        print(f"\n  연속 {consecutive_failures}회 실패. {FAILURE_PAUSE_SECONDS}초 대기...")
                        time.sleep(FAILURE_PAUSE_SECONDS)
                        consecutive_failures = 0
                time.sleep(2)

            if ta_frames:
                ta_all = pd.concat(ta_frames, ignore_index=True)
                ta_grid = (
                    ta_all.groupby(['ny', 'nx', 'lat', 'lon'], as_index=False)['value']
                    .mean()
                    .rename(columns={'value': 'ta'})
                )
                ta_grid['ta'] = ta_grid['ta'].round(4)
            else:
                ta_grid = None

            # rn_day: 격자별 단일 값
            rn_df = fetch_haenam_grid_data(date_str + "0000", 'rn_day', AUTH_KEY)
            if rn_df is not None:
                rn_grid = rn_df.rename(columns={'value': 'rn_day'})
                rn_grid['rn_day'] = rn_grid['rn_day'].round(4)
                consecutive_failures = 0
            else:
                rn_grid = None
                consecutive_failures += 1
            time.sleep(2)

            # ta와 rn_day 격자 병합
            if ta_grid is not None and rn_grid is not None:
                day_df = pd.merge(
                    ta_grid,
                    rn_grid[['ny', 'nx', 'rn_day']],
                    on=['ny', 'nx'], how='outer'
                )
            elif ta_grid is not None:
                day_df = ta_grid.copy()
                day_df['rn_day'] = ''
            elif rn_grid is not None:
                day_df = rn_grid.copy()
                day_df['ta'] = ''
            else:
                # 둘 다 실패 → 기록하지 않고 다음 실행 때 재시도
                print("ta=N/A, rn_day=N/A (재시도 예정)")
                continue

            day_df.insert(0, 'date', date_str)
            day_df = day_df[['date', 'ny', 'nx', 'lat', 'lon', 'ta', 'rn_day']]
            day_df.to_csv(OUTPUT_FILE, index=False, mode='a', header=False, encoding='utf-8-sig')

            n_grids = len(day_df)
            ta_str = f"OK ({len(ta_frames)}/6시간)" if ta_grid is not None else "N/A"
            rn_str = "OK" if rn_grid is not None else "N/A"
            print(f"{n_grids}개 격자, ta={ta_str}, rn_day={rn_str}")

    except KeyboardInterrupt:
        print("\n\n중단됨. 다시 실행하면 마지막 저장 지점부터 이어서 처리합니다.")

    print(f"\n완료! 결과 파일: {OUTPUT_FILE}")
