"""爆量黑 K 後做多（短期反轉 / 提供流動性）回測。

文獻依據
- 進場：Campbell, Grossman & Wang (1993, QJE)：指數在大量時的報酬之後會反轉；
  Nagel (2012, RFS)：短期反轉 = 提供流動性的報酬，VIX 高時較大。
  本研究事件研究（volume_spike.py）：相對量 ≥ 3 的黑 K，之後到收盤平均 +7.7bp，2016–2026 每年皆正。
- 不設停損：Kaminski & Lo (2014)：價格呈反轉特性時，停損會降低報酬。
- 部位：純 VIX（2% / VIX 隱含單日波動，上限 4 倍），與 A 版相同；另以固定 1 倍對照。

規則（回測前事先定好）
- 正規盤 08:31–14:30 CT（台北 21:31–03:30 夏令 / 22:31–04:30 冬令）每分鐘檢查；
  相對量（該分鐘量 ÷ 過去 14 天同一分鐘平均量）≥ 3 且收盤 < 開盤 → 以該 K 棒收盤（或延遲 delay 分鐘）買進。
- 同一時間只持有一筆；空手時同一天可再進場。
- 出場：'close' 抱到 15:00 CT；'vwap' 回到當日 VWAP 即停利，否則抱到收盤；'30m' 持有 30 分鐘；
  'next' 抱到隔天 15:00 CT。
"""
import numpy as np
import pandas as pd
import intraday_mom as m
import multiday as md          # 共用 C、VWAP、VIX 對照表與統計函式

TARGET_VOL = 0.02
es = md.es
t = es.index.time
rth = es[(t >= pd.Timestamp("08:30").time()) & (t <= pd.Timestamp("15:00").time())]
day, tod = rth.index.normalize(), rth.index.strftime("%H:%M")
piv = lambda s: s.to_frame("v").assign(day=day, tod=tod).pivot(index="day", columns="tod", values="v")
C = md.C
Op = piv(rth.Open).reindex(index=C.index, columns=C.columns)
V = piv(rth.Vol.astype(float)).reindex(index=C.index, columns=C.columns).fillna(0)
relv = V / V.rolling(14).mean().shift(1)
cols = list(C.columns)
j0, j1, jc = cols.index("08:31"), cols.index("14:30"), cols.index("15:00")
days = [d for d in md.days if d <= pd.Timestamp("2026-08-13")]      # VIX 資料到 2026-08-14
Ca, Oa, Ra, Wa, Xa = (x.loc[days].to_numpy() for x in (C, Op, relv, md.vwap, md.vix_at))
Ca = C.loc[days].to_numpy()


def backtest(exit="close", th=3.0, delay=0, cost=m.COST_PTS, sizing="vix", lev_cap=m.MAX_LEV):
    n = len(days)
    daily = np.zeros(n)
    trades = []
    carry = None                                   # 'next' 出場用：(k, mark, entry_px, entry_i, entry_j)
    for i in range(n):
        c, o, r, w, vx = Ca[i], Oa[i], Ra[i], Wa[i], Xa[i]
        o0 = c[0]
        pnl = 0.0
        if carry is not None:                      # 前一天留倉 → 今天收盤出場
            k, mark, epx, ei, ej = carry
            x = c[jc]
            pnl += k * (x - mark) - k * cost
            trades.append((days[ei], cols[ej], days[i], "15:00", k * (x - epx) - 2 * k * cost))
            carry = None
            daily[i] = pnl
            continue                               # 出場當天不再進場（簡化）
        j = j0
        while j <= j1:
            if not (r[j] >= th and c[j] < o[j]):
                j += 1
                continue
            je = min(j + delay, jc)
            v = vx[j]
            if sizing == "vix":
                if np.isnan(v):
                    j += 1
                    continue
                L = min(TARGET_VOL / (v / 100 / np.sqrt(252)), lev_cap)
            else:
                L = 1.0
            epx = c[je]
            k = L / epx
            pnl -= k * cost
            if exit == "next":
                pnl += k * (c[jc] - epx)
                carry = (k, c[jc], epx, i, je)
                break
            if exit == "close":
                jx = jc
            elif exit == "30m":
                jx = min(je + 30, jc)
            else:                                  # 'vwap'
                hit = np.nonzero(c[je + 1:jc + 1] >= w[je + 1:jc + 1])[0]
                jx = je + 1 + hit[0] if len(hit) else jc
            x = c[jx]
            pnl += k * (x - epx) - k * cost
            trades.append((days[i], cols[je], days[i], cols[jx], k * (x - epx) - 2 * k * cost))
            j = jx + 1
        daily[i] = pnl
    t = pd.DataFrame(trades, columns=["entry_day", "t_in", "exit_day", "t_out", "ret"])
    return pd.Series(daily, index=pd.DatetimeIndex(days)), t


if __name__ == "__main__":
    bh = md.bh.loc[days[0]:days[-1]]
    R, T = {"ES 買進持有": bh}, {}
    for lab, kw in [("抱到收盤", dict(exit="close")), ("回到VWAP停利", dict(exit="vwap")),
                    ("持有30分", dict(exit="30m")), ("抱到隔天收盤", dict(exit="next")),
                    ("抱到收盤｜固定1倍", dict(exit="close", sizing="const")),
                    ("抱到收盤｜延遲1分", dict(exit="close", delay=1)),
                    ("抱到收盤｜2 tick", dict(exit="close", cost=0.5)),
                    ("抱到收盤｜延遲1分＋2tick＋槓桿≤2", dict(exit="close", delay=1, cost=0.5, lev_cap=2))]:
        R[lab], T[lab] = backtest(**kw)
    R = pd.DataFrame(R)
    print(f"樣本 {R.index[0].date()} ~ {R.index[-1].date()}（只做多；樣本外 = 2019-11-22 以前）\n")
    print(md.summarize(R, T))
    print("\n每年 Sharpe：")
    print(R.groupby(R.index.year).apply(lambda x: x.mean() / x.std() * np.sqrt(252)).T.round(2).to_string())

    # 依前一日 VIX 分組（固定 1 倍，排除部位大小影響）— 檢驗 Nagel (2012)：VIX 高時反轉報酬較大
    t = T["抱到收盤｜固定1倍"].copy()
    t["vix"] = t.entry_day.map(md.vix_at["15:00"].shift(1))
    g = t.groupby(pd.cut(t.vix, [0, 15, 20, 30, 100]), observed=True).ret
    print("\n依前一日 VIX 分組（固定 1 倍，每筆 bp）：")
    print(pd.DataFrame({"筆數": g.size(), "平均bp": g.mean() * 1e4, "勝率": g.apply(lambda x: (x > 0).mean())}).round(2).to_string())

    # 與 A 版的關係
    A, _ = md.backtest(base="dvwap", ratchet=False, sizing="vix")
    A = A.reindex(R.index)
    rev = R["抱到收盤"]
    sr = lambda x: x.mean() / x.std() * np.sqrt(252)
    mdd = lambda x: ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min()
    print(f"\n與 A 版日報酬相關：{A.corr(rev):.2f}")
    for lab, x in [("A 版", A), ("爆量黑K·抱到收盤", rev), ("各半", 0.5 * A + 0.5 * rev), ("兩者全額相加", A + rev)]:
        print(f"  {lab:12s} Sharpe {sr(x):.2f}  年化 {x.mean()*252:.1%}  MDD {mdd(x):.1%}  "
              f"2016–19 {sr(x[:'2019-11-21']):.2f}  2019-11後 {sr(x['2019-11-22':]):.2f}  2025後 {sr(x['2025':]):.2f}")
    R.to_pickle("results/vol_reversal.pkl")
