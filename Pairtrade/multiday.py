"""多日版動能策略：突破雜訊區間進場、VWAP 停損、分段上移停損、VIX 決定部位。

文獻依據
- 進場：Zarattini, Aziz & Barbon (2024) 的雜訊區間（Noise Area）突破；只在整點/半點檢查（09:00–14:30 CT）。
  另加「價格 ≥ 當日 VWAP」。只做多（本研究實證：空單無優勢）。
- 停損基準（兩種都測）：
  · 每日 VWAP：論文的 VWAP 移動停損（論文為 max(UB, VWAP)；Q22 測過只用 VWAP）。
  · 進場起算 VWAP（Anchored VWAP，錨定在進場當天 08:30 CT 開盤，之後跨日累積、含夜盤成交量）：
    實務做法，讓部位能持有多天。進場當天等同每日 VWAP，之後不再每日重設。
    （若錨定在進場那一根 K 棒，VWAP 起始值 = 進場價，等於立刻保本停損，實測幾乎每筆都當天被洗出。）
- 分段上移停損：第 k 層 = 進場價 × (1 + k·s)，s = 進場當下 VIX/100/√252（VIX 隱含的單日波動）。
  價格碰到第 k 層（k ≥ 1）後，停損至少上移到第 k−1 層（第 0 層 = 進場價，即保本），只升不降。
  間接依據：Kaminski & Lo (2014)，動能環境下停損規則有正貢獻。
- 部位：槓桿 = min(上限, 2% / (VIX/100/√252))，進場時固定。依據 Moreira & Muir (2017)、
  Harvey et al. (2018)；論文 4.1 節亦建議依當下波動動態調整部位。
- 持有：沒有觸發停損就一直持有，不設最長天數。

停損只在正規盤的檢查時點判斷（夜盤不判斷，跳空在隔天第一個檢查點處理）。
資料以台北時間儲存，這裡用芝加哥時間（ct=True）判斷交易時段。
"""
import numpy as np
import pandas as pd
from load import load
import intraday_mom as m

TARGET_VOL = 0.02
es = load("ES", ct=True)
vix = load("VIX", ct=True).Close
C, O, ub, lb, sigma, lev14, twap, vwap = m.prep(es)
lev14_raw = TARGET_VOL / C["15:00"].pct_change().rolling(m.LOOKBACK).std().shift(1)   # 未設上限的論文槓桿
days = [d for d in C.index if not np.isnan(lev14[d]) and not np.isnan(sigma.loc[d, "09:00"])]
entry_t = [c for c in C.columns if c.endswith((":00", ":30")) and "09:00" <= c <= "14:30"]

# 進場起算 VWAP 用：全時段（含夜盤）的累積 價×量 與 量
typ = ((es.High + es.Low + es.Close) / 3).to_numpy()
vol = es.Vol.to_numpy(dtype=float)
cum_pv, cum_v = np.cumsum(typ * vol), np.cumsum(vol)
# 每個 (交易日, 分鐘) → 全時段資料的位置（取該分鐘或之前最後一根）
ts = (pd.DatetimeIndex(np.repeat(C.index.values, len(C.columns)))
      + pd.to_timedelta(np.tile([c + ":00" for c in C.columns], len(C.index))))
gidx = pd.DataFrame(es.index.searchsorted(ts, side="right").reshape(len(C.index), -1) - 1,
                    index=C.index, columns=C.columns)
vix_pos = vix.index.searchsorted(ts, side="right").reshape(len(C.index), -1) - 1
# 只用 15 分鐘內的 VIX 報價，避免資料結束後沿用舊值（VIX 資料到 2026-08-14，ES 到 2026-10）
fresh = (vix_pos >= 0) & (ts.values.reshape(len(C.index), -1) - vix.index.values[np.clip(vix_pos, 0, None)]
                          <= np.timedelta64(15, "m"))
vix_at = pd.DataFrame(np.where(fresh, vix.to_numpy()[np.clip(vix_pos, 0, None)], np.nan),
                      index=C.index, columns=C.columns)


def stop_times(freq):
    return [c for c in C.columns if int(c[3:]) % freq == 0 and "08:35" <= c <= "15:00" and (freq < 30 or c >= "09:00")]


def backtest(base="avwap", ratchet=True, freq=5, sizing="vix", lev_cap=m.MAX_LEV, cost=m.COST_PTS, delay=0, step=1.0):
    """base: 'avwap'（進場起算）或 'dvwap'（每日）。
    sizing: 'vix'（2% / VIX 隱含單日波動）、'rv14'（論文：2% / 過去 14 日報酬標準差）、
            'blend'（2% / VIX 與 RV14 的平均；Busch et al. 2011）、'const'（固定 1 倍）。
    step: 分段間隔 = step × VIX 隱含單日波動。"""
    st = set(stop_times(freq))
    times = sorted(st | set(entry_t) | {"15:00"})
    X = C.shift(-delay, axis=1).ffill(axis=1) if delay else C
    pos, k = 0, 0.0
    daily, trades = {}, []
    for d in days:
        pnl = 0.0
        Cd, Xd, ud, vd, gd, xd = C.loc[d], X.loc[d], ub.loc[d], vwap.loc[d], gidx.loc[d], vix_at.loc[d]
        for c in times:
            p, x = Cd[c], Xd[c]
            if pos:
                pnl += k * (x - mark)
                mark = x
                if c in st:
                    if base == "avwap":
                        g = gd[c]
                        line = (cum_pv[g] - cum_pv[g0 - 1]) / max(cum_v[g] - cum_v[g0 - 1], 1)
                    else:
                        line = vd[c]
                    if ratchet:
                        lvl = max(lvl, int(np.floor((p / entry_px - 1) / s)))
                        if lvl >= 1:
                            line = max(line, entry_px * (1 + (lvl - 1) * s))
                    if p < line:
                        pnl -= k * cost
                        trades.append((entry_day, d, c, k * (x - entry_px) - 2 * k * cost, lvl if ratchet else 0))
                        pos = 0
                continue
            if c in entry_t and p > ud[c] and p >= vd[c]:
                v = xd[c]
                if np.isnan(v) and sizing in ("vix", "blend"):
                    continue
                iv = v / 100 / np.sqrt(252)                      # VIX 隱含單日波動
                rv = TARGET_VOL / lev14_raw[d]                   # 過去 14 日報酬標準差
                L = min({"vix": TARGET_VOL / iv if sizing == "vix" else 0, "rv14": TARGET_VOL / rv,
                         "blend": TARGET_VOL / ((iv + rv) / 2) if sizing == "blend" else 0,
                         "const": 1.0}[sizing], lev_cap)
                s = step * (v if not np.isnan(v) else 20.0) / 100 / np.sqrt(252)   # 分段間隔
                pos, k, entry_px, mark, entry_day = 1, L / O[d], x, x, d
                g0, lvl = gd["08:30"], 0          # 錨定在進場當天開盤
                pnl -= k * cost
        daily[d] = pnl
    t = pd.DataFrame(trades, columns=["entry_day", "exit_day", "t_out", "ret", "max_lvl"])
    return pd.Series(daily), t


bh = C["15:00"].pct_change().reindex(days)
OOS = pd.Timestamp("2019-11-22")
sr = lambda x: x.mean() / x.std() * np.sqrt(252)
mdd = lambda x: ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min()
cagr = lambda x: (1 + x).prod() ** (252 / len(x)) - 1


def summarize(R, T):
    rows = []
    for k in R:
        r = R[k].dropna()
        row = {"策略": k, "年化報酬": cagr(r), "年化波動": r.std() * np.sqrt(252), "Sharpe": sr(r), "最大回撤": mdd(r),
               "樣本外": sr(r[r.index < OOS]), "2019-11後": sr(r[r.index >= OOS]), "2025後": sr(r[r.index >= "2025-01-01"])}
        t = T.get(k)
        if t is not None:
            hold = np.busday_count(t.entry_day.values.astype("datetime64[D]"), t.exit_day.values.astype("datetime64[D]"))
            row.update(筆數=len(t), 勝率=(t.ret > 0).mean(), 平均持有日=hold.mean(), 最長持有日=hold.max(),
                       在場比例=(r != 0).mean())
        rows.append(row)
    pc, f2 = "{:.1%}".format, "{:.2f}".format
    return pd.DataFrame(rows).set_index("策略").to_string(formatters={
        "年化報酬": pc, "年化波動": pc, "最大回撤": pc, "Sharpe": f2, "樣本外": f2, "2019-11後": f2, "2025後": f2,
        "勝率": "{:.0%}".format, "平均持有日": "{:.1f}".format, "最長持有日": "{:.0f}".format,
        "筆數": "{:.0f}".format, "在場比例": "{:.0%}".format})


if __name__ == "__main__":
    cases = [
        ("每日VWAP停損（對照，VIX部位）", dict(base="dvwap", ratchet=False)),
        ("每日VWAP＋分段上移", dict(base="dvwap", ratchet=True)),
        ("進場起算VWAP停損", dict(base="avwap", ratchet=False)),
        ("進場起算VWAP＋分段上移", dict(base="avwap", ratchet=True)),
        ("進場起算VWAP＋分段上移·30分檢查", dict(base="avwap", ratchet=True, freq=30)),
        ("進場起算VWAP＋分段上移·論文部位(RV14)", dict(base="avwap", ratchet=True, sizing="rv14")),
    ]
    R, T = {"ES 買進持有": bh}, {}
    for lab, kw in cases:
        R[lab], T[lab] = backtest(**kw)
    R = pd.DataFrame(R)
    print(f"樣本 {R.index[0].date()} ~ {R.index[-1].date()}（只做多；樣本外 = 2019-11-22 以前）\n")
    print(summarize(R, T))
    print("\n每年 Sharpe：")
    print(R.groupby(R.index.year).apply(lambda x: x.mean() / x.std() * np.sqrt(252)).T.round(2).to_string())
    R.to_pickle("results/multiday.pkl")
    pd.to_pickle(T, "results/multiday_trades.pkl")
