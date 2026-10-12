"""資料讀取：1 分鐘 CSV（新到舊）→ 排序、時區轉換、換月調整 → pickle 快取。

原始 CSV 留在上一層的 Pairtrade/（檔案很大，不複製）；快取存在 final/cache/。

所有資料統一以台北時間表示。美股交易時段的規則（正規盤 08:30–15:00 CT）在台北時間
會隨美國夏令時間跳動（夏令 21:30–04:00、冬令 22:30–05:00，且跨午夜），
所以分析時用 load(name, ct=True) 暫時轉成芝加哥交易所時間，輸出時再用 to_taipei() 轉回。
"""
from pathlib import Path
import numpy as np
import pandas as pd

CSV_DIR = Path(__file__).resolve().parent.parent      # Pairtrade/
CACHE_DIR = Path(__file__).resolve().parent / "cache"
FILES = {"ES": "ES #F Chart 2025-09-23-10-05.csv",   # 含成交量，2016-12 起；取代舊的 ES #F_1min.csv（價格完全相同）
         "VIX": "$VIX 1m.csv", "NKD": "NKD #F_1min.csv"}
SRC_TZ = {"ES": "Asia/Taipei", "VIX": "America/Chicago", "NKD": "America/Chicago"}   # 各檔案原始時間戳記的時區
TPE, CT = "Asia/Taipei", "America/Chicago"
ROLLS = {"ES", "NKD"}   # 未經換月調整的連續期貨合約
ES_EXPIRY_HOLIDAYS = {pd.Timestamp("2026-06-19")}   # 第 3 個週五遇美股休市


def _convert(idx: pd.DatetimeIndex, src: str, dst: str) -> pd.DatetimeIndex:
    # 美國秋季回撥那一小時在芝加哥時間有歧義 → 設為 NaT 後丟掉（每年最多 60 根）
    return idx.tz_localize(src, ambiguous="NaT", nonexistent="NaT").tz_convert(dst).tz_localize(None)


def _third_friday(y, m):
    d = pd.Timestamp(y, m, 15)
    return d + pd.Timedelta(days=(4 - d.weekday()) % 7)


def roll_dates(name: str, start, end) -> list:
    """連續合約換月日（芝加哥日期；換月發生在當天 17:00 CT 重新開盤時）。
    ES 與 NKD 相同：3/6/9/12 月第 3 個週五往前 8 天的週四（NKD 即第 2 個週五的前一天）。
    ES 到期日若遇美股休市（2026-06-19 Juneteenth）提前一天，換月日也跟著提前。
    已用 2016–2026 每季的休市跳空驗證。"""
    out = []
    for y in range(start.year, end.year + 1):
        for m in (3, 6, 9, 12):
            exp = _third_friday(y, m)
            if name == "ES" and exp in ES_EXPIRY_HOLIDAYS:
                exp -= pd.Timedelta(days=1)
            out.append(exp - pd.Timedelta(days=8))
    return [d for d in out if start <= d <= end]


def back_adjust(df_ct: pd.DataFrame, name: str) -> tuple[pd.DataFrame, pd.Series]:
    """以比例向後調整換月跳空（最新價格不變）。df_ct 為芝加哥時間。回傳 (調整後資料, 各次跳空 log)。

    跳空 = 換月日 17:00 後第一筆開盤 / 17:00 前最後一筆收盤，包含維護時段內真實的市場變動
    （一般日中位數 ES 約 0.02%、NKD 約 0.05%），可忽略。"""
    idx = df_ct.index
    factor = np.ones(len(df_ct))
    gaps = {}
    for d in roll_dates(name, idx[0].normalize(), idx[-1].normalize()):
        i = idx.searchsorted(d + pd.Timedelta(hours=17))
        if i == 0 or i == len(idx):
            continue
        g = df_ct.Open.iat[i] / df_ct.Close.iat[i - 1]
        gaps[d] = np.log(g)
        factor[:i] *= g
    df = df_ct.copy()
    px = ["Open", "High", "Low", "Close"]
    df[px] = df[px].to_numpy() * factor[:, None]
    return df, pd.Series(gaps, name="roll_gap")


def load(name: str, ct: bool = False, adjust: bool = True) -> pd.DataFrame:
    """回傳台北時間的 1 分鐘資料；ct=True 時改為芝加哥時間（判斷交易時段用）。
    adjust=True（預設）時 ES/NKD 價格已做換月向後調整；adjust=False 為原始報價。"""
    cache = CACHE_DIR / f"{name}_tpe.pkl"
    if cache.exists():
        df = pd.read_pickle(cache)
    else:
        df = pd.read_csv(CSV_DIR / FILES[name])
        tm = df.pop("Time")
        if tm.str.contains("午").any():
            tm = tm.str[3:] + np.where(tm.str.startswith("上午"), " AM", " PM")   # 去掉「上午 」/「下午 」
            fmt = "%Y/%m/%d %I:%M:%S %p"
        else:
            fmt = "%Y/%m/%d %H:%M:%S"
        df.index = pd.to_datetime(df.pop("Date") + " " + tm, format=fmt)
        if SRC_TZ[name] != TPE:
            df.index = _convert(df.index, SRC_TZ[name], TPE)
        df = df[df.index.notna() & ~df.index.duplicated()].sort_index()
        cache.parent.mkdir(exist_ok=True)
        df.to_pickle(cache)
    if not ct and not (adjust and name in ROLLS):
        return df
    df = df.copy()
    df.index = _convert(df.index, TPE, CT)
    df = df[df.index.notna() & ~df.index.duplicated()]
    if adjust and name in ROLLS:
        df = back_adjust(df, name)[0]
    if not ct:
        df.index = _convert(df.index, CT, TPE)
    return df


def tpe_clock(hhmm: str) -> str:
    """芝加哥時鐘 'HH:MM' → 台北時鐘 '夏令/冬令'，例如 '09:00' → '22:00/23:00'。"""
    h, m = map(int, hhmm.split(":"))
    return f"{(h + 13) % 24:02d}:{m:02d}/{(h + 14) % 24:02d}:{m:02d}"


def to_taipei(ts) -> pd.DatetimeIndex:
    """芝加哥時間（無時區）→ 台北時間（無時區）。"""
    return _convert(pd.DatetimeIndex(ts), CT, TPE)


if __name__ == "__main__":
    for n in FILES:
        d = load(n)
        print(n, "台北時間", d.index[0], "~", d.index[-1], len(d))
        print(d.groupby(d.index.hour).size().to_string())
