import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def setup():
    candidates = [
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf',
        '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    ]
    for path in candidates:
        if os.path.exists(path):
            fe = fm.FontEntry(fname=path, name=os.path.splitext(os.path.basename(path))[0])
            fm.fontManager.ttflist.append(fe)
            plt.rcParams['font.family'] = fe.name
            plt.rcParams['axes.unicode_minus'] = False
            return
    print("[WARNING] 한글 폰트를 찾을 수 없습니다. sudo apt-get install -y fonts-nanum 실행 후 ~/.cache/matplotlib 삭제하세요.")
