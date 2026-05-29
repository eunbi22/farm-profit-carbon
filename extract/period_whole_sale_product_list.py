"""
KAMIS API 16번: 신)일별 품목별 도매 가격자료
대상: 쌀 (부류코드: 100, 품목코드: 111)
"""

import json
import requests
from datetime import datetime, timedelta

API_KEY = "b55b4819-6b83-446b-92a9-46da1ac84082"
CERT_ID = "8175"

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


if __name__ == "__main__":
    today         = datetime.today()
    one_month_ago = today - timedelta(days=30)

    params = {
        **PARAMS_RICE,
        "p_startday": one_month_ago.strftime("%Y-%m-%d"),
        "p_endday":   today.strftime("%Y-%m-%d"),
    }

    resp = requests.get(BASE_URL, params=params, timeout=15)
    resp.raise_for_status()

    data = resp.json()

    out_path = "rice_wholesale.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"저장 완료: {out_path}")
