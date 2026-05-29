"""
KAMIS API 16번: 신)일별 품목별 도매 가격자료
대상: 쌀 (부류코드: 100, 품목코드: 111)
"""

import ssl
import json
import requests
import pandas as pd
import urllib3
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── 인증 정보 ──────────────────────────────────────────────
API_KEY = "b55b4819-6b83-446b-92a9-46da1ac84082"
CERT_ID = "8175"

# ── 요청 파라미터 ──────────────────────────────────────────
BASE_URL = "http://www.kamis.or.kr/service/price/xml.do"

PARAMS_RICE = {
    "action": "periodWholesaleProductList",
    "p_cert_key": API_KEY,
    "p_cert_id": CERT_ID,
    "p_returntype": "json",
    "p_itemcategorycode": "100",
    "p_itemcode": "111",
    "p_kindcode": "01",
    "p_productrankcode": "04",
    "p_countrycode": "2401",
    "p_convert_kg_yn": "Y",
}


class LegacySSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.options |= ssl.OP_LEGACY_SERVER_CONNECT
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        kwargs["ssl_context"] = ctx
        super().init_poolmanager(*args, **kwargs)


def fetch_kamis_wholesale(start_day: str, end_day: str, params: dict = PARAMS_RICE) -> pd.DataFrame:
    p = {**params, "p_startday": start_day, "p_endday": end_day}

    session = requests.Session()
    session.mount("https://", LegacySSLAdapter())
    session.mount("http://", LegacySSLAdapter())

    resp = session.get(BASE_URL, params=p, timeout=15, verify=False)
    resp.raise_for_status()

    data = resp.json()

    if "data" not in data:
        print("응답 원문:", json.dumps(data, ensure_ascii=False, indent=2))
        raise ValueError("data 키 없음 - 응답 구조 확인 필요")

    items = data["data"].get("item", [])
    if not items:
        print("조회 결과 없음:", json.dumps(data, ensure_ascii=False, indent=2))
        return pd.DataFrame()

    return pd.DataFrame(items)


def fetch_yearly_data(year: int) -> pd.DataFrame:
    start = f"{year}-01-01"
    end   = f"{year}-12-31"
    print(f"  {year}년 수집 중... ({start} ~ {end})")
    return fetch_kamis_wholesale(start, end)


def clean_price_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    print("원본 컬럼:", df.columns.tolist())
    print(df.head(3).to_string())

    date_col  = next((c for c in df.columns if "date" in c.lower() or "날짜" in c or "일자" in c), None)
    price_col = next((c for c in df.columns if "price" in c.lower() or "가격" in c), None)

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.rename(columns={date_col: "date"})

    if price_col:
        df[price_col] = pd.to_numeric(df[price_col].astype(str).str.replace(",", ""), errors="coerce")
        df = df.rename(columns={price_col: "price_per_kg"})

    df = df.dropna(subset=["date"] if "date" in df.columns else [])
    df = df.sort_values("date").reset_index(drop=True)

    return df


if __name__ == "__main__":
    today         = datetime.today()
    one_month_ago = today - timedelta(days=30)

    print("=" * 50)
    print("KAMIS API 16번 - 쌀 도매가격 테스트")
    print(f"조회 기간: {one_month_ago.strftime('%Y-%m-%d')} ~ {today.strftime('%Y-%m-%d')}")
    print(f"지역: 광주(2401) | 품종: 20kg | 등급: 상품(04)")
    print("=" * 50)

    df = fetch_kamis_wholesale(
        start_day=one_month_ago.strftime("%Y-%m-%d"),
        end_day=today.strftime("%Y-%m-%d")
    )

    if not df.empty:
        df = clean_price_df(df)
        print(f"\n수집 완료: {len(df)}건")
        print(df.to_string())

        out_path = "rice_wholesale_test.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\nCSV 저장: {out_path}")

    # ── 장기 데이터 수집 (연도별 루프) ────────────────────────
    # years = [2020, 2021, 2022, 2023, 2024]
    # dfs = []
    # for y in years:
    #     try:
    #         df_y = fetch_yearly_data(y)
    #         if not df_y.empty:
    #             dfs.append(df_y)
    #     except Exception as e:
    #         print(f"  {y}년 오류: {e}")
    #
    # if dfs:
    #     df_all = pd.concat(dfs, ignore_index=True)
    #     df_all = clean_price_df(df_all)
    #     df_all.to_csv("rice_wholesale_2020_2024.csv", index=False, encoding="utf-8-sig")
    #     print(f"전체 수집: {len(df_all)}건")