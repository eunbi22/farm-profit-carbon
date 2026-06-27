"""
Batch-fetch soil characteristics from the Korean Open API for every PNU in the DBF.

Source : ../data/LSMD_CONT_LDREG_46820_202605.dbf  (EUC-KR, 445 421 rows)
Output : ../data/토양특성_추출결과.csv               (UTF-8-sig)

Checkpoint: results are flushed to CSV every CHECKPOINT_INTERVAL rows so a crash
            will not lose more than that many records.
"""

import csv
import os
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dbfread import DBF
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent.parent
DBF_PATH   = BASE_DIR / "data" / "LSMD_CONT_LDREG_46820_202605.dbf"
OUTPUT_CSV = BASE_DIR / "data" / "토양특성_추출결과.csv"

API_URL      = "http://apis.data.go.kr/1390802/SoilEnviron/SoilCharac/V3/getSoilCharacter"
SERVICE_KEY  = "33d9104b9f2d0d3f88919927bc3a886afa6e277b161f5567b49e2cfa92dc2e42"

MAX_WORKERS          = 15
CHECKPOINT_INTERVAL  = 2_000
REQUEST_TIMEOUT      = 15   # seconds per request
RETRY_TOTAL          = 3
RETRY_BACKOFF        = 0.5  # seconds

OUTPUT_COLUMNS = ["PNU", "PNU_Cd", "Soildra_Cd", "Vldsoildep_Cd", "Surtture_Cd"]

# ---------------------------------------------------------------------------
# HTTP session factory (one per thread via thread-local storage)
# ---------------------------------------------------------------------------
import threading
_thread_local = threading.local()

def _get_session() -> requests.Session:
    """Return a thread-local requests.Session with retry logic."""
    if not hasattr(_thread_local, "session"):
        retry = Retry(
            total=RETRY_TOTAL,
            backoff_factor=RETRY_BACKOFF,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        s = requests.Session()
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        _thread_local.session = s
    return _thread_local.session


# ---------------------------------------------------------------------------
# API call & XML parse
# ---------------------------------------------------------------------------
def fetch_soil(pnu: str) -> dict:
    """
    Call the API for one PNU and return a flat dict with the 3 target fields.
    All missing / error values are stored as empty string.
    """
    result = {col: "" for col in OUTPUT_COLUMNS}
    result["PNU"] = pnu

    try:
        session = _get_session()
        resp = session.get(
            API_URL,
            params={"serviceKey": SERVICE_KEY, "PNU_CD": pnu},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()

        root = ET.fromstring(resp.content)

        # The API wraps data in <items><item>…</item></items>
        item = root.find(".//item")
        if item is None:
            return result  # no data row for this PNU

        def _text(tag: str) -> str:
            el = item.find(tag)
            if el is None or el.text is None:
                return ""
            return el.text.strip()

        result["PNU_Cd"]        = _text("PNU_Cd")
        result["Soildra_Cd"]    = _text("Soildra_Cd")
        result["Vldsoildep_Cd"] = _text("Vldsoildep_Cd")
        result["Surtture_Cd"]   = _text("Surtture_Cd")

    except ET.ParseError:
        pass  # malformed XML → leave fields empty
    except requests.RequestException:
        pass  # network / HTTP error → leave fields empty

    return result


# ---------------------------------------------------------------------------
# DBF loading: collect valid 19-digit PNU strings
# ---------------------------------------------------------------------------
def load_pnu_list() -> list[str]:
    print(f"[INFO] Reading DBF: {DBF_PATH}")
    table = DBF(str(DBF_PATH), encoding="euc-kr", load=False)

    pnu_list = []
    for record in table:
        raw = record.get("PNU", "")
        # Force to plain string to avoid scientific notation from numeric types
        pnu_str = str(raw).strip() if raw is not None else ""
        # Strip any decimal artefacts (e.g. "4682011234567890123.0")
        if "." in pnu_str:
            pnu_str = pnu_str.split(".")[0]
        if len(pnu_str) == 19 and pnu_str.isdigit():
            pnu_list.append(pnu_str)

    print(f"[INFO] Valid 19-digit PNUs found: {len(pnu_list):,}")
    return pnu_list


# ---------------------------------------------------------------------------
# Checkpoint writer
# ---------------------------------------------------------------------------
class CheckpointWriter:
    def __init__(self, path: Path, columns: list[str]):
        self.path    = path
        self.columns = columns
        self._buffer: list[dict] = []
        self._total_written = 0

        # Write header if file is new / empty
        write_header = not path.exists() or path.stat().st_size == 0
        self._fh = open(path, "a", newline="", encoding="utf-8-sig")
        self._writer = csv.DictWriter(self._fh, fieldnames=columns)
        if write_header:
            self._writer.writeheader()

    def add(self, row: dict):
        self._buffer.append(row)
        if len(self._buffer) >= CHECKPOINT_INTERVAL:
            self.flush()

    def flush(self):
        if self._buffer:
            self._writer.writerows(self._buffer)
            self._fh.flush()
            self._total_written += len(self._buffer)
            print(f"[CHECKPOINT] {self._total_written:,} rows written to CSV")
            self._buffer.clear()

    def close(self):
        self.flush()
        self._fh.close()

    @property
    def total(self):
        return self._total_written + len(self._buffer)


# ---------------------------------------------------------------------------
# Resume support: skip PNUs already present in output file
# ---------------------------------------------------------------------------
def load_done_pnus(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    done = set()
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pnu = row.get("PNU", "").strip()
            if pnu:
                done.add(pnu)
    print(f"[INFO] Resuming — {len(done):,} PNUs already in output, skipping.")
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pnu_list = load_pnu_list()

    done_pnus = load_done_pnus(OUTPUT_CSV)
    todo      = [p for p in pnu_list if p not in done_pnus]
    print(f"[INFO] PNUs to fetch: {len(todo):,}")

    if not todo:
        print("[INFO] Nothing to do. Output is already complete.")
        return

    writer    = CheckpointWriter(OUTPUT_CSV, OUTPUT_COLUMNS)
    completed = 0
    errors    = 0
    start     = time.time()

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_soil, pnu): pnu for pnu in todo}

        for future in as_completed(futures):
            try:
                row = future.result()
                writer.add(row)
            except Exception:
                errors += 1

            completed += 1
            if completed % 500 == 0:
                elapsed  = time.time() - start
                rate     = completed / elapsed if elapsed > 0 else 0
                eta_secs = (len(todo) - completed) / rate if rate > 0 else 0
                print(
                    f"[PROGRESS] {completed:,}/{len(todo):,} "
                    f"({completed/len(todo)*100:.1f}%) | "
                    f"{rate:.1f} req/s | "
                    f"ETA {eta_secs/60:.1f} min | "
                    f"errors: {errors}"
                )

    writer.close()
    elapsed = time.time() - start
    print(f"\n[DONE] Total written: {writer.total:,} rows in {elapsed/60:.1f} min")
    print(f"[DONE] Output: {OUTPUT_CSV}")
    if errors:
        print(f"[WARN] {errors} futures raised unexpected exceptions.")


if __name__ == "__main__":
    main()
