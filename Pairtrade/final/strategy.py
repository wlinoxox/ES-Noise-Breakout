"""ES 雜訊區間突破策略（最終版本）。

整合自研究腳本 intraday_mom.py（雜訊區間）、multiday.py（VIX 部位、留倉）、a_overnight.py（夜盤停損）。
規則與文獻依據見同資料夾的 README.md。

時間：資料以台北時間儲存，這裡用芝加哥時間（CT）判斷交易時段，避免美國夏令時間錯位。
  正規盤 08:30–15:00 CT = 台北 21:30–04:00（美國夏令）/ 22:30–05:00（美國冬令）。
"""
import numpy as np
import pandas as pd
from data import load

# ---- 參數 ----
LOOKBACK = 14          # 雜訊區間 σ 與 14 日實現波動的回看天數（論文）
TARGET_VOL = 0.02      # 目標單日波動（論文）
MAX_LEV = 4.0          # 槓桿上限（論文）
COST_PTS = 0.25        # 每邊成本 = 1 tick（含手續費與滑價）
STOP_FREQ = 5          # 正規盤停損檢查頻率（分鐘）
NIGHT_FREQ = 5         # 夜盤停損檢查頻率（分鐘）
VIX_MAX_AGE = 15       # 進場時 VIX 報價最多可以是幾分鐘前
VIX_END = pd.Timestamp("2026-08-13")   # VIX 資料到 2026-08-14，回測只到前一個完整交易日

# ---- 資料 ----
es = load("ES", ct=True)               # 已做換月向後調整
vix = load("VIX", ct=True).Close
_t = es.index.time
rth = es[(_t >= pd.Timestamp("08:30").time()) & (_t <= pd.Timestamp("15:00").time())]
_day, _tod = rth.index.normalize(), rth.index.strftime("%H:%M")


def _piv(s):
    return s.to_frame("v").assign(day=_day, tod=_tod).pivot(index="day", columns="tod", values="v")


C = _piv(rth.Close).ffill(axis=1)                       # 交易日 × 分鐘 收盤價
O = rth.Open.groupby(_day).first().reindex(C.index)     # 08:30 開盤價
C = C[C.notna().sum(axis=1) > 300]                      # 去掉假日縮短交易、提前收盤日
O = O.reindex(C.index)
prev_close = C["15:00"].shift(1)

# 雜訊區間（Zarattini, Aziz & Barbon 2024）
sigma = (C.div(O, axis=0) - 1).abs().rolling(LOOKBACK).mean().shift(1)    # 只用前 14 天
_ref_up = np.maximum(O, prev_close)
UB = sigma.mul(_ref_up, axis=0).add(_ref_up, axis=0)

# 每日 VWAP（典型價格 (H+L+C)/3 以成交量加權，從 08:30 累積）
_typ = _piv((rth.High + rth.Low + rth.Close) / 3).reindex(C.index).ffill(axis=1)
_vol = _piv(rth.Vol.astype(float)).reindex(C.index).fillna(0)
_twap = _typ.cumsum(axis=1) / _typ.notna().cumsum(axis=1)
VWAP = ((_typ * _vol).cumsum(axis=1) / _vol.cumsum(axis=1)).ffill(axis=1).fillna(_twap)

# 論文的 14 日實現波動槓桿（未設上限），只在比較部位算法時用到
STD14 = C["15:00"].pct_change().rolling(LOOKBACK).std().shift(1)

# 每個 (交易日, 分鐘) 當下可用的 VIX（只接受 VIX_MAX_AGE 分鐘內的報價）
_ts = (pd.DatetimeIndex(np.repeat(C.index.values, len(C.columns)))
       + pd.to_timedelta(np.tile([c + ":00" for c in C.columns], len(C.index))))
_vp = vix.index.searchsorted(_ts, side="right") - 1
_fresh = (_vp >= 0) & ((_ts.values - vix.index.values[np.clip(_vp, 0, None)]) <= np.timedelta64(VIX_MAX_AGE, "m"))
VIX = pd.DataFrame(np.where(_fresh, vix.to_numpy()[np.clip(_vp, 0, None)], np.nan).reshape(len(C.index), -1),
                   index=C.index, columns=C.columns)

DAYS = [d for d in C.index if not np.isnan(STD14[d]) and not np.isnan(sigma.loc[d, "09:00"]) and d <= VIX_END]
ENTRY_T = [c for c in C.columns if c.endswith((":00", ":30")) and "09:00" <= c <= "14:30"]

# 夜盤區間：(交易日 15:00 CT, 下一個交易日 08:30 CT)
_idx = es.index
_cl = es.Close.to_numpy()
_mins = (_idx.hour * 60 + _idx.minute).to_numpy()
_lo = _idx.searchsorted([d + pd.Timedelta(hours=15) for d in DAYS], side="right")
_hi = np.r_[_idx.searchsorted([d + pd.Timedelta(hours=8, minutes=30) for d in DAYS[1:]], side="left"), _lo[-1]]


def stop_times(freq):
    return [c for c in C.columns if int(c[3:]) % freq == 0 and "08:35" <= c <= "15:00" and (freq < 30 or c >= "09:00")]


def leverage(sizing, vix_now, day, lev_cap):
    """sizing：'vix'（最終版本）、'rv14'（論文）、'blend'（VIX 與 RV14 平均）、'const'（固定 1 倍）。"""
    iv = vix_now / 100 / np.sqrt(252)            # VIX 隱含單日波動
    rv = STD14[day]
    L = {"vix": TARGET_VOL / iv, "rv14": TARGET_VOL / rv, "blend": TARGET_VOL / ((iv + rv) / 2), "const": 1.0}[sizing]
    return min(L, lev_cap)


def backtest(night="wide", sizing="vix", cost=COST_PTS, lev_cap=MAX_LEV, freq=STOP_FREQ, nfreq=NIGHT_FREQ):
    """回傳 (每日報酬 Series, 逐筆交易 DataFrame)。

    night：夜盤停損線。'wide'（最終版本）= 前一天 VWAP × (1 − VIX/100/√252)；
           'vwap' = 前一天 VWAP；None = 夜盤不停損。
    報酬為部位損益 ÷ 帳戶淨值（名目 = 槓桿 × 淨值，以當天 08:30 開盤價換算口數）。
    """
    st = set(stop_times(freq))
    times = sorted(st | set(ENTRY_T) | {"15:00"})
    nmask = _mins % nfreq == 0
    need_vix = sizing in ("vix", "blend")
    pos, k = 0, 0.0
    daily, trades = np.zeros(len(DAYS)), []
    for i, d in enumerate(DAYS):
        pnl = 0.0
        # 1) 前一晚的夜盤停損
        if pos and night is not None and i > 0:
            a, b = _lo[i - 1], _hi[i - 1]
            seg = np.arange(a, b)[nmask[a:b]]
            line = last_vwap * (1 - last_vix / 100 / np.sqrt(252)) if night == "wide" else last_vwap
            hit = np.nonzero(_cl[seg] < line)[0]
            if len(hit):
                x = _cl[seg[hit[0]]]
                pnl += k * (x - mark) - k * cost
                trades.append((entry_day, entry_t, d, "夜盤", k * (x - epx) - 2 * k * cost,
                               epx, x, L, vix_in, _idx[seg[hit[0]]]))
                pos = 0
        # 2) 正規盤：停損、進場
        Cd, Ud, Wd, Xd = C.loc[d], UB.loc[d], VWAP.loc[d], VIX.loc[d]
        for c in times:
            p = Cd[c]
            if pos:
                pnl += k * (p - mark)
                mark = p
                if c in st and p < Wd[c]:
                    pnl -= k * cost
                    trades.append((entry_day, entry_t, d, c, k * (p - epx) - 2 * k * cost,
                                   epx, p, L, vix_in, d + pd.Timedelta(c + ":00")))
                    pos = 0
                continue
            if c in ENTRY_T and p > Ud[c] and p >= Wd[c] and not (need_vix and np.isnan(Xd[c])):
                L = leverage(sizing, Xd[c], d, lev_cap)
                pos, k, epx, mark, entry_day, entry_t, vix_in = 1, L / O[d], p, p, d, c, Xd[c]
                pnl -= k * cost
        # 3) 收盤仍持有 → 留倉；記下夜盤停損要用的 VWAP 與 VIX
        if pos:
            last_vwap = Wd["15:00"]
            v = Xd.dropna()
            last_vix = v.iloc[-1] if len(v) else 20.0
        daily[i] = pnl
    # px_in / px_out 為換月向後調整後的價格；exit_ct 為出場時間（芝加哥時間）
    t = pd.DataFrame(trades, columns=["entry_day", "t_in", "exit_day", "t_out", "ret",
                                      "px_in", "px_out", "lev", "vix_in", "exit_ct"])
    return pd.Series(daily, index=pd.DatetimeIndex(DAYS)), t


buy_hold = C["15:00"].pct_change().reindex(DAYS)
OOS_END = pd.Timestamp("2019-11-22")      # 之前的資料在研究後期才取得 → 樣本外


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    sr = lambda x: float(x.mean() / x.std() * np.sqrt(252))
    eq = (1 + r).cumprod()
    return {"cagr": float(eq.iloc[-1] ** (252 / len(r)) - 1), "vol": float(r.std() * np.sqrt(252)),
            "sharpe": sr(r), "mdd": float((eq / eq.cummax() - 1).min()),
            "sh_oos": sr(r[r.index < OOS_END]), "sh_is": sr(r[r.index >= OOS_END]),
            "sh_2025": sr(r[r.index >= "2025-01-01"]), "worst_day": float(r.min())}
