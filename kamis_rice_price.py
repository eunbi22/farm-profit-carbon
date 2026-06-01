import requests
import pandas as pd
import json

CERT_KEY = "b55b4819-6b83-446b-92a9-46da1ac84082"
CERT_ID = "8175"
BASE_URL = "https://www.kamis.or.kr/service/price/xml.do"

def fetch_yearly_price(start_year: int, end_year: int) -> list[dict]:
    params = {
        "action": "yearlyPriceTrendList",
        "p_cert_key": CERT_KEY,
        "p_cert_id": CERT_ID,
        "p_returntype": "json",
        "p_itemcategorycode": "100",
        "p_itemcode": "111",
        "p_kindcode": "01",
        "p_productrankcode": "04",
        "p_countrycode": "1101",
        "p_convert_kg_yn": "N",
        "p_startday": str(start_year),
        "p_endday": str(end_year),
    }

    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()

    body = resp.json()
    print("응답 구조 확인:", json.dumps(body, ensure_ascii=False, indent=2)[:500])

    data = body.get("data", body)
    if isinstance(data, dict) and data.get("error_code") not in (None, "000", ""):
        raise ValueError(f"API 오류: {data.get('error_code')} - {data.get('error_message', '')}")

    # 항목 추출 (API 응답 구조에 따라 조정)
    items = []
    if isinstance(data, dict):
        for key in ("item", "price", "list", "data"):
            if key in data:
                raw = data[key]
                items = raw if isinstance(raw, list) else [raw]
                break
    elif isinstance(data, list):
        items = data

    return items


def main():
    print("KAMIS 쌀 연도별 도매가격 수집 시작 (2020~2025)")
    rows = fetch_yearly_price(2020, 2025)

    if not rows:
        print("수집된 데이터가 없습니다. API 응답 구조를 확인하세요.")
        return

    df = pd.DataFrame(rows)
    print(f"\n수집 완료: {len(df)}행")
    print(df.head(10).to_string())

    out_path = "rice_yearly_price_2020_2025.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
