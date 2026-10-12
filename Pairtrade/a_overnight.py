"""A 版 + 夜盤停損：正規盤規則完全不變，只在留倉過夜期間（15:00 CT → 隔天 08:30 CT，
台北 04:00 → 21:30 夏令 / 05:00 → 22:30 冬令）加上停損檢查。

夜盤停損線（事先定好三種）
- 'prev_vwap'：前一天正規盤收盤時的 VWAP（即 A 版決定留倉的那條線）
- 'wide'    ：前一天正規盤 VWAP × (1 − VIX 隱含單日波動)，只防大跌
- 'globex'  ：17:00 CT 夜盤開盤後重新累積的 VWAP（17:00 前沿用前一天 VWAP）；對照組
"""
import numpy as np
import pandas as pd
import intraday_mom as m
import multiday as md

TARGET_VOL = 0.02
es = md.es
idx = es.index
cl = es.Close.to_numpy()
tp = ((es.High + es.Low + es.Close) / 3).to_numpy()
vo = es.Vol.to_numpy(dtype=float)
mins = (idx.hour * 60 + idx.minute).to_numpy()
days = [d for d in md.days if d <= pd.Timestamp("2026-08-13")]
C, UB, VW, VX = (x.loc[days] for x in (md.C, md.ub, md.vwap, md.vix_at))
cols = list(C.columns)
entry_t = set(md.entry_t)
# 每個交易日之後的夜盤區間：(當天 15:00, 下一個交易日 08:30)
lo = idx.searchsorted([d + pd.Timedelta(hours=15) for d in days], side="right")
hi = np.r_[idx.searchsorted([d + pd.Timedelta(hours=8, minutes=30) for d in days[1:]], side="left"), lo[-1]]


def backtest(night=None, nfreq=5, freq=5, cost=m.COST_PTS, lev_cap=m.MAX_LEV):
    st = set(md.stop_times(freq))
    times = sorted(st | entry_t | {"15:00"})
    nmask = (mins % nfreq == 0)
    pos, k = 0, 0.0
    daily, trades, night_exits = np.zeros(len(days)), [], 0
    for i, d in enumerate(days):
        pnl = 0.0
        if pos and night is not None and i > 0:         # 檢查前一晚的夜盤
            a, b = lo[i - 1], hi[i - 1]
            seg = np.arange(a, b)[nmask[a:b]]
            if len(seg):
                if night == "prev_vwap":
                    line = np.full(len(seg), last_vwap)
                elif night == "wide":
                    line = np.full(len(seg), last_vwap * (1 - last_vix / 100 / np.sqrt(252)))
                else:                                   # globex：17:00 起重新累積
                    j = np.arange(a, b)
                    after = mins[j] >= 17 * 60
                    after |= np.cumsum(after) > 0       # 跨過午夜仍屬同一段
                    pv = np.cumsum(np.where(after, tp[j] * vo[j], 0))
                    vv = np.cumsum(np.where(after, vo[j], 0))
                    gv = np.where(vv > 0, pv / np.maximum(vv, 1), last_vwap)
                    line = gv[nmask[a:b]]
                hit = np.nonzero(cl[seg] < line)[0]
                if len(hit):
                    x = cl[seg[hit[0]]]
                    pnl += k * (x - mark) - k * cost
                    trades.append((entry_day, d, "夜盤", k * (x - epx) - 2 * k * cost))
                    pos, night_exits = 0, night_exits + 1
        Cd, Ud, Wd, Xd = C.iloc[i], UB.iloc[i], VW.iloc[i], VX.iloc[i]
        for c in times:
            p = Cd[c]
            if pos:
                pnl += k * (p - mark)
                mark = p
                if c in st and p < Wd[c]:
                    pnl -= k * cost
                    trades.append((entry_day, d, c, k * (p - epx) - 2 * k * cost))
                    pos = 0
                continue
            if c in entry_t and p > Ud[c] and p >= Wd[c] and not np.isnan(Xd[c]):
                L = min(TARGET_VOL / (Xd[c] / 100 / np.sqrt(252)), lev_cap)
                pos, k, epx, mark, entry_day = 1, L / md.O[d], p, p, d
                pnl -= k * cost
        if pos:
            last_vwap = Wd["15:00"]
            v = Xd.dropna()
            last_vix = v.iloc[-1] if len(v) else 20.0
        daily[i] = pnl
    t = pd.DataFrame(trades, columns=["entry_day", "exit_day", "t_out", "ret"])
    return pd.Series(daily, index=pd.DatetimeIndex(days)), t, night_exits


if __name__ == "__main__":
    R, T, N = {"ES 買進持有": md.bh.loc[days[0]:days[-1]]}, {}, {}
    cases = [("A 版（夜盤不停損）", dict()),
             ("夜盤：跌破前一天 VWAP·5分", dict(night="prev_vwap")),
             ("夜盤：跌破前一天 VWAP·30分", dict(night="prev_vwap", nfreq=30)),
             ("夜盤：寬停損（VWAP−1σ）·5分", dict(night="wide")),
             ("夜盤：寬停損（VWAP−1σ）·30分", dict(night="wide", nfreq=30)),
             ("夜盤：夜盤重新累積 VWAP·5分", dict(night="globex"))]
    for lab, kw in cases:
        R[lab], T[lab], N[lab] = backtest(**kw)
        if lab != "A 版（夜盤不停損）":
            R[lab + "｜2 tick"], T[lab + "｜2 tick"], N[lab + "｜2 tick"] = backtest(cost=0.5, **kw)
    R["A 版（夜盤不停損）｜2 tick"], T["A 版（夜盤不停損）｜2 tick"], _ = backtest(cost=0.5)
    R = pd.DataFrame(R).loc["2016-12-22":]
    print(f"樣本 {R.index[0].date()} ~ {R.index[-1].date()}\n")
    print(md.summarize(R, T))
    print("\n夜盤停損出場次數：", {k: v for k, v in N.items() if "2 tick" not in k})
    # 最差單日：夜盤停損是否真的降低尾部風險
    print("\n最差 5 天（%）：")
    print(R[[c for c in R if "2 tick" not in c]].apply(lambda x: x.nsmallest(5).mul(100).round(2).tolist()).to_string())
