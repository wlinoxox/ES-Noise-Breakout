"""只做多的日內動能，停損改為「跌破 VWAP」（另以 TWAP 對照）。

進場：同原版，每半小時（09:00–14:30 CT，台北 22:00–03:30 夏令 / 23:00–04:30 冬令）價格 > 上緣 UB 且 ≥ 當日均價線時做多。
出場：價格 < 當日均價線（VWAP 或 TWAP）才停損，否則持續持有。
- 當沖版：15:00 一律平倉
- 留倉版：15:00 若仍 ≥ 均價線就抱過夜；隔天用新一天的均價線檢查
停損檢查頻率：每 30 分鐘或每 5 分鐘。部位大小在進場時依當日槓桿固定，直到出場。
"""
import numpy as np
import pandas as pd
from load import load
import intraday_mom as m

es = load("ES", ct=True)
C, O, ub, lb, sigma, lev, twap, vwap = m.prep(es)
days = [d for d in C.index if not np.isnan(lev[d]) and not np.isnan(sigma.loc[d, "09:00"])]
entry_t = [c for c in C.columns if c.endswith((":00", ":30")) and "09:00" <= c <= "14:30"]


def stop_times(freq):
    return [c for c in C.columns if int(c[3:]) % freq == 0 and "08:35" <= c <= "15:00" and (freq < 30 or c >= "09:00")]


def backtest(freq=30, carry=False, stop="twap", delay=0, cost=m.COST_PTS, lev_cap=m.MAX_LEV):
    """回傳 (日報酬, 逐筆交易)。stop='vwap'/'twap' 只看均價線；'band' 為原版（跌回 UB 以下）；'both' = max(UB, VWAP)（論文版）。

    訊號用 K 棒 c 的收盤判斷；成交價為 delay 分鐘後那根 K 棒的收盤（delay=0 即同一根，略為樂觀）。
    cost 為單邊點數；lev_cap 為槓桿上限。"""
    avg = twap if stop == "twap" else vwap
    X = C.shift(-delay, axis=1).ffill(axis=1) if delay else C     # 成交 / 盯市價格
    st = stop_times(freq)
    times = sorted(set(st) | set(entry_t) | {"15:00"})
    pos, k, entry_px, entry_day, mark = 0, 0.0, 0.0, None, 0.0
    daily, trades = {}, []
    for d in days:
        pnl = 0.0
        Cd, Xd, ud, td = C.loc[d], X.loc[d], ub.loc[d], avg.loc[d]
        for c in times:
            p, x = Cd[c], Xd[c]
            if pos and c in st:
                line = {"twap": td[c], "vwap": td[c], "band": ud[c], "both": max(ud[c], td[c])}[stop]
                last = c == "15:00"
                if p < line or (last and not carry):
                    pnl += k * (x - mark) - k * cost
                    trades.append((entry_day, d, c, k * (x - entry_px) - 2 * k * cost))
                    pos = 0
                    continue
            if pos:
                pnl += k * (x - mark)
                mark = x
            elif c in entry_t and p > ud[c] and p >= td[c]:
                pos, k, entry_px, mark, entry_day = 1, min(lev[d], lev_cap) / O[d], x, x, d
                pnl -= k * cost
        if pos and not carry:                    # 保險：當沖版不留倉
            pnl += k * (Xd["15:00"] - mark) - k * cost
            pos = 0
        daily[d] = pnl
    return pd.Series(daily), pd.DataFrame(trades, columns=["entry_day", "exit_day", "t_out", "ret"])


bh = C["15:00"].pct_change().reindex(days)


if __name__ == "__main__":
    R, T = {"ES 買進持有": bh}, {}
    for lab, kw in [("原版停損（跌回UB）·30分", dict(stop="band")),
                    ("max(UB,VWAP)·30分（論文版）", dict(stop="both")),
                    ("TWAP 停損·5分·可留倉（上次最佳）", dict(stop="twap", freq=5, carry=True)),
                    ("VWAP 停損·30分·當沖", dict(stop="vwap")),
                    ("VWAP 停損·5分·當沖", dict(stop="vwap", freq=5)),
                    ("VWAP 停損·30分·可留倉", dict(stop="vwap", carry=True)),
                    ("VWAP 停損·5分·可留倉", dict(stop="vwap", freq=5, carry=True))]:
        R[lab], T[lab] = backtest(**kw)
    R = pd.DataFrame(R).dropna()

    rows = []
    for k in R:
        s = m.stats(R[k], k)
        t = T.get(k)
        if t is not None:
            hold = (t.exit_day - t.entry_day).dt.days
            s.update(筆數=len(t), 勝率=(t.ret > 0).mean(), 平均持有天數=hold.mean())
        rows.append(s)
    print(f"樣本 {R.index[0].date()} ~ {R.index[-1].date()}（只做多）\n")
    print(pd.DataFrame(rows).set_index("策略").to_string(formatters={
        "年化報酬": "{:.1%}".format, "年化波動": "{:.1%}".format, "Sharpe": "{:.2f}".format, "最大回撤": "{:.1%}".format,
        "交易日占比": "{:.0%}".format, "勝率": "{:.0%}".format, "平均持有天數": "{:.2f}".format, "筆數": "{:.0f}".format}))
    print("\n每年 Sharpe：")
    print(R.groupby(R.index.year).apply(lambda x: x.mean() / x.std() * np.sqrt(252)).T.round(2).to_string())

    # 2019-11-22 以前的資料是這次新增的，上次挑規則時沒看過 → 視為樣本外
    OOS = pd.Timestamp("2019-11-22")
    sr = lambda x: x.mean() / x.std() * np.sqrt(252)
    mdd = lambda x: ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min()
    print("\n樣本外（2016-12 ~ 2019-11）vs 原樣本（2019-11 之後）：")
    print(pd.DataFrame({"Sharpe 樣本外": R[R.index < OOS].apply(sr), "MDD 樣本外": R[R.index < OOS].apply(mdd),
                        "Sharpe 原樣本": R[R.index >= OOS].apply(sr), "MDD 原樣本": R[R.index >= OOS].apply(mdd)}).to_string(
        formatters={"Sharpe 樣本外": "{:.2f}".format, "Sharpe 原樣本": "{:.2f}".format, "MDD 樣本外": "{:.1%}".format, "MDD 原樣本": "{:.1%}".format}))
    R.to_pickle("results/mom_vwap.pkl")
