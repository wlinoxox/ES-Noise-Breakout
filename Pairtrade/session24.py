"""全天候版 A 策略：以 CME 電子盤交易日（17:00 CT 開盤 → 隔天 16:00 CT）為單位。

台北時間：06:00 → 隔天 05:00（美國夏令）；07:00 → 隔天 06:00（冬令）。

規則（把論文的正規盤定義延伸到整個電子盤時段）
- 開盤價 = 17:00 CT 開盤；σ(t) = 過去 14 個交易日「從開盤到第 t 分鐘」|報酬| 的平均；
  上緣 = max(開盤價, 前一交易日 15:59 收盤) × (1 + σ(t))。
- VWAP 從 17:00 CT 開始累積（含夜盤）。
- 進場：開盤後 30 分鐘到收盤前 30 分鐘（17:30–15:30 CT），每半小時檢查；價格 > 上緣且 ≥ VWAP → 做多。
- 停損：全時段每 freq 分鐘檢查，價格 < VWAP 出場。
- 留倉：15:59 CT 仍 ≥ VWAP 就跨過維護時段留到下一交易日，改用新交易日的 VWAP。
- 部位：槓桿 = min(4, 2% / (VIX/100/√252))；VIX 只在 02:15–15:15 CT 有報價，夜盤沿用最近一筆。

延伸依據：論文只研究正規盤；Elaut, Frömmel & Lampaert (2018) 在 24 小時交易的外匯市場也發現交易時段動能，
Baltussen et al. (2021) 在多種期貨上發現日內動能。
"""
import numpy as np
import pandas as pd
from load import load
import intraday_mom as m

TARGET_VOL, LOOKBACK = 0.02, 14
OPEN_MIN = 17 * 60                                      # 17:00 CT
es = load("ES", ct=True)
vix = load("VIX", ct=True).Close

# 交易日標籤：17:00 CT 之後的 K 棒屬於隔天（+7 小時後的日期）
sess = (es.index + pd.Timedelta(hours=7)).normalize()
mos = ((es.index.hour * 60 + es.index.minute - OPEN_MIN) % 1440).to_numpy()   # 開盤後第幾分鐘
ok = mos < 23 * 60                                      # 排除 16:00–16:59 維護時段的零星資料
es, sess, mos = es[ok], sess[ok], mos[ok]
piv = lambda s: pd.DataFrame({"v": s.to_numpy(), "d": sess, "k": mos}).pivot(index="d", columns="k", values="v")
Craw = piv(es.Close)
nbar = Craw.notna().sum(axis=1)
keep = nbar > 1200                                      # 排除假日縮短交易日（完整約 1380 分鐘）
Craw = Craw[keep].reindex(columns=range(1380))
C = Craw.ffill(axis=1)
O = piv(es.Open)[keep].reindex(columns=range(1380)).bfill(axis=1).iloc[:, 0]   # 開盤後第一筆
V = piv(es.Vol.astype(float))[keep].reindex(columns=range(1380)).fillna(0)
TP = piv((es.High + es.Low + es.Close) / 3)[keep].reindex(columns=range(1380)).ffill(axis=1)
VW = ((TP * V).cumsum(axis=1) / V.cumsum(axis=1)).ffill(axis=1).fillna(TP)
close = C[1379]                                         # 15:59 CT
prev_close = close.shift(1)
sigma = (C.div(O, axis=0) - 1).abs().rolling(LOOKBACK).mean().shift(1)
ref = np.maximum(O, prev_close)
UB = sigma.mul(ref, axis=0).add(ref, axis=0)

# 每個 (交易日, 分鐘) 的 VIX：最近一筆（夜盤沿用前一天收盤），最多回看 4 天（週末），VIX 資料結束後為 NaN
ts = (pd.DatetimeIndex(np.repeat(C.index.values, 1380)) - pd.Timedelta(hours=7)
      + pd.to_timedelta(np.tile(np.arange(1380) + OPEN_MIN, len(C)), unit="m"))
vp = vix.index.searchsorted(ts, side="right") - 1
fresh = (vp >= 0) & ((ts.values - vix.index.values[np.clip(vp, 0, None)]) <= np.timedelta64(4, "D"))
VX = np.where(fresh, vix.to_numpy()[np.clip(vp, 0, None)], np.nan).reshape(len(C), 1380)
VX[np.asarray(C.index > pd.Timestamp("2026-08-13"))] = np.nan

days = [d for d in C.index if not sigma.loc[d].isna().all() and d <= pd.Timestamp("2026-08-13")]
di = C.index.get_indexer(days)
Ca, Ua, Wa = C.to_numpy()[di], UB.to_numpy()[di], VW.to_numpy()[di]
VXa = VX[di]
ENTRY = np.arange(30, 1380 - 29, 30)                    # 17:30 … 15:30 CT


def label(k):
    t = (OPEN_MIN + k) % 1440
    return f"{t // 60:02d}:{t % 60:02d}"


def backtest(freq=5, cost=m.COST_PTS, lev_cap=m.MAX_LEV, delay=0, entry_mask=None, stop_mask=None):
    """entry_mask / stop_mask：長度 1380 的布林陣列，限制可進場 / 可檢查停損的分鐘（預設全時段）。"""
    stops = np.zeros(1380, bool)
    stops[freq::freq] = True
    stops[1379] = True
    if stop_mask is not None:
        stops &= stop_mask
        stops[1379] = True
    ent = np.zeros(1380, bool)
    ent[ENTRY] = True
    if entry_mask is not None:
        ent &= entry_mask
    times = np.nonzero(stops | ent)[0]
    pos, k = 0, 0.0
    daily, trades = np.zeros(len(days)), []
    for i in range(len(days)):
        c, u, w, vx = Ca[i], Ua[i], Wa[i], VXa[i]
        pnl = 0.0
        for j in times:
            p = c[j]
            x = c[min(j + delay, 1379)]
            if pos:
                pnl += k * (x - mark)
                mark = x
                if stops[j] and (p < w[j]):
                    pnl -= k * cost
                    trades.append((days[e_i], label(e_j), days[i], label(j), k * (x - epx) - 2 * k * cost))
                    pos = 0
                continue
            if ent[j] and p > u[j] and p >= w[j] and not np.isnan(vx[j]):
                L = min(TARGET_VOL / (vx[j] / 100 / np.sqrt(252)), lev_cap)
                pos, k, epx, mark, e_i, e_j = 1, L / x, x, x, i, j
                pnl -= k * cost
        daily[i] = pnl
    t = pd.DataFrame(trades, columns=["entry_day", "t_in", "exit_day", "t_out", "ret"])
    return pd.Series(daily, index=pd.DatetimeIndex(days)), t


bh = close.pct_change().reindex(days)

if __name__ == "__main__":
    import multiday as md
    A, TA = md.backtest(base="dvwap", ratchet=False, sizing="vix")
    rth = np.zeros(1380, bool)
    rth[15 * 60 + 30: 22 * 60 + 1] = True               # 08:30–15:00 CT
    cases = [("全天候：進場＋停損都全時段", {}),
             ("正規盤進場、停損全時段", dict(entry_mask=rth)),
             ("全時段進場、停損只在正規盤", dict(stop_mask=rth)),
             ("全天候｜延遲1分", dict(delay=1)),
             ("全天候｜2 tick", dict(cost=0.5)),
             ("全天候｜槓桿≤2", dict(lev_cap=2)),
             ("全天候｜保守：延遲1分＋2tick＋槓桿≤2", dict(delay=1, cost=0.5, lev_cap=2))]
    R, T = {"ES 買進持有": bh, "A 版（正規盤）": A.reindex(days)}, {"A 版（正規盤）": TA}
    for lab, kw in cases:
        R[lab], T[lab] = backtest(**kw)
    R = pd.DataFrame(R).loc["2016-12-22":]
    print(f"樣本 {R.index[0].date()} ~ {R.index[-1].date()}（只做多；樣本外 = 2019-11-22 以前）\n")
    print(md.summarize(R, T))
    print("\n每年 Sharpe：")
    print(R.groupby(R.index.year).apply(lambda x: x.mean() / x.std() * np.sqrt(252)).T.round(2).to_string())
    t = T["全天候：進場＋停損都全時段"]
    hr = t.t_in.str[:2].astype(int)
    seg = np.select([(hr >= 17) | (hr < 2), (hr >= 2) & (hr < 8), (hr >= 8) & (hr < 15)], ["亞洲 17–02 CT", "歐洲 02–08 CT", "美國 08–15 CT"], "收盤前 15–16 CT")
    g = t.groupby(seg).ret
    print("\n全天候版：依進場時段（台北：亞洲 06–15、歐洲 15–21、美國 21–04，夏令）")
    print(pd.DataFrame({"筆數": g.size(), "平均bp": g.mean() * 1e4, "勝率": g.apply(lambda x: (x > 0).mean()),
                        "總報酬%": g.sum() * 100}).round(2).to_string())
    R.to_pickle("results/session24.pkl")
