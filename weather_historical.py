import os
import io
import time
import requests
import xarray as xr
import pandas as pd
import numpy as np

AUTH_KEY = "5uVnBqgbS8KlZwaoG8vC8w"
OUTPUT_FILE = "haenam_weather_historical.csv"

LAT_MIN, LAT_MAX = 34.28, 34.67
LON_MIN, LON_MAX = 126.24, 126.75

# 4시간 간격 6지점 (00, 04, 08, 12, 16, 20시)
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
            lat_res = requests.get(lat_url, timeout=90)
            lon_res = requests.get(lon_url, timeout=90)
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


def fetch_haenam_mean(tm_str, obs_element, auth_key, retries=3):
    """특정 시각의 요소 데이터를 받아 해남군 격자 평균값 반환. 실패 시 None."""
    url = (
        f"https://apihub.kma.go.kr/api/typ01/url/sfc_grid_nc_down.php"
        f"?obs={obs_element}&tm={tm_str}&authKey={auth_key}"
    )
    temp_nc = f"_tmp_{obs_element}_{tm_str}.nc"

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code != 200:
                print(f"  HTTP {resp.status_code} [{tm_str}|{obs_element}] (시도 {attempt}/{retries})")
                time.sleep(2 * attempt)
                continue

            with open(temp_nc, "wb") as f:
                f.write(resp.content)

            with xr.open_dataset(temp_nc) as ds:
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
            ]

            if haenam.empty:
                print(f"  해남 격자 없음 [{tm_str}|{obs_element}]")
                return None

            return float(haenam['value'].mean())

        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"  오류 [{tm_str}|{obs_element}] (시도 {attempt}/{retries}): {e}")
            time.sleep(2 * attempt)

        finally:
            if os.path.exists(temp_nc):
                os.remove(temp_nc)

    return None


if __name__ == "__main__":
    dates = pd.date_range(start="2020-05-02", end="2025-12-31", freq="D")

    # 이미 처리된 날짜 확인 (중단 후 재시작 지원)
    done_dates = set()
    if os.path.exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE, dtype=str)
        done_dates = set(existing['date'].tolist())
        print(f"기존 파일 발견: {len(done_dates)}일 이미 처리됨\n")
    else:
        pd.DataFrame(columns=['date', 'ta', 'rn_day']).to_csv(
            OUTPUT_FILE, index=False, encoding='utf-8-sig'
        )

    # 위경도 맵 미리 로드
    get_latlon_map(AUTH_KEY)

    total = len(dates)
    try:
        for i, date in enumerate(dates, 1):
            date_str = date.strftime("%Y%m%d")

            if date_str in done_dates:
                continue

            print(f"[{i}/{total}] {date_str} 처리 중...", end=" ", flush=True)

            # ta: 4시간 간격 6회 수집 후 평균
            ta_values = []
            for hour in TA_HOURS:
                val = fetch_haenam_mean(date_str + hour, 'ta', AUTH_KEY)
                if val is not None:
                    ta_values.append(val)
                time.sleep(0.5)

            ta_mean = round(float(np.mean(ta_values)), 4) if ta_values else None

            # rn_day: 00시 일일 강수량 1회
            rn_mean = fetch_haenam_mean(date_str + "0000", 'rn_day', AUTH_KEY)
            rn_mean = round(rn_mean, 4) if rn_mean is not None else None
            time.sleep(0.5)

            row = pd.DataFrame([{
                'date': date_str,
                'ta': ta_mean if ta_mean is not None else '',
                'rn_day': rn_mean if rn_mean is not None else '',
            }])
            row.to_csv(OUTPUT_FILE, index=False, mode='a', header=False, encoding='utf-8-sig')

            ta_str = f"{ta_mean:.2f}°C ({len(ta_values)}/6)" if ta_mean is not None else "N/A"
            rn_str = f"{rn_mean}mm" if rn_mean is not None else "N/A"
            print(f"ta={ta_str}, rn_day={rn_str}")

    except KeyboardInterrupt:
        print("\n\n중단됨. 다시 실행하면 마지막 저장 지점부터 이어서 처리합니다.")

    print(f"\n완료! 결과 파일: {OUTPUT_FILE}")
