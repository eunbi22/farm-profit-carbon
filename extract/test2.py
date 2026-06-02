import json

# 파일 경로를 본인의 환경에 맞게 수정하세요
import json

# 확장자를 .geojson으로 바꾸고, 경로를 확실하게 한 단계 위(..)의 data 폴더로 지정합니다.
with open('../data/haenam_farmmap_20260527_1312.geojson', 'r', encoding='utf-8') as f:
    data = json.load(f)
    # 첫 번째 feature의 구조만 출력
    print(json.dumps(data['features'][0], indent=2, ensure_ascii=False))
    data = json.load(f)
    # 첫 번째 feature의 구조만 출력
    print(json.dumps(data['features'][0], indent=2, ensure_ascii=False))