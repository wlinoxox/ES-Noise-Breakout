"""Zarattini, Aziz & Barbon (2024) "Beat the Market" 日內動能策略，套用到 ES 期貨，並測試 VIX 濾網。

規則（依論文；無成交量資料，故不使用 VWAP 停損，只用雜訊區間邊界）：
- σ(t) = 過去 14 天同一分鐘 |Close_t / Open_day − 1| 的平均
- UB = max(Open, 前日收盤)·(1+σ)，LB = min(Open, 前日收盤)·(1−σ)
- 每半小時（09:00–14:30 CT，台北 22:00–03:30 夏令 / 23:00–04:30 冬令）檢查：價格 > UB 做多、< LB 做空；持多時跌回 UB 以下平倉（空單對稱）
- 15:00 CT（台北 04:00/05:00）全部平倉；槓桿 = min(4, 2% / 過去 14 日日報酬標準差)
- 成本：每次單邊 0.25 點（約 1 tick，含手續費與滑價）
資料以台北時間儲存，交易時段用芝加哥時間判斷（load(..., ct=True)），避免夏令時間錯位。
"""
import numpy as np
import pandas as pd
from load import load

LOOKBACK, TARGET_VOL, MAX_LEV, COST_PTS = 14, 0.02, 4.0, 0.25


def stats(r: pd.Series, label: str) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    return {"策略": label, "年化報酬": eq.iloc[-1] ** (1 / yrs) - 1, "年化波動": r.std() * np.sqrt(252),
            "Sharpe": r.mean() / r.std() * np.sqrt(252), "最大回撤": (eq / eq.cummax() - 1).min(),
            "交易日占比": (r != 0).mean()}


def prep(es: pd.DataFrame):
    """把正規盤 1 分鐘資料轉成 日×分鐘 矩陣，並算出雜訊區間與槓桿。"""
    t = es.index.time
    rth = es[(t >= pd.Timestamp("08:30").time()) & (t <= pd.Timestamp("15:00").time())]
    day, tod = rth.index.normalize(), rth.index.strftime("%H:%M")
    piv = lambda s: s.to_frame("v").assign(day=day, tod=tod).pivot(index="day", columns="tod", values="v")
    C = piv(rth.Close).ffill(axis=1)
    O = rth.Open.groupby(day).first().reindex(C.index)
    C = C[C.notna().sum(axis=1) > 300]          # 去掉提前收盤/資料不全的日子
    O = O.reindex(C.index)
    prev_close = C["15:00"].shift(1)

    move = (C.div(O, axis=0) - 1).abs()
    sigma = move.rolling(LOOKBACK).mean().shift(1)            # 只用前 14 天
    ub = sigma.mul(np.maximum(O, prev_close), axis=0).add(np.maximum(O, prev_close), axis=0)
    lb = np.minimum(O, prev_close).to_frame().values * (1 - sigma)
    daily_ret = C["15:00"].pct_change()
    lev = (TARGET_VOL / daily_ret.rolling(LOOKBACK).std().shift(1)).clip(upper=MAX_LEV)
    # 典型價格 (H+L+C)/3 的當日累積平均：TWAP 等權，VWAP 以成交量加權
    typ = piv((rth.High + rth.Low + rth.Close) / 3).reindex(C.index)
    vol = piv(rth.Vol.astype(float)).reindex(C.index).fillna(0)
    typ = typ.ffill(axis=1)
    twap = typ.cumsum(axis=1) / typ.notna().cumsum(axis=1)
    vwap = ((typ * vol).cumsum(axis=1) / vol.cumsum(axis=1)).ffill(axis=1).fillna(twap)
    return C, O, ub, lb, sigma, lev, twap, vwap


def run(es: pd.DataFrame, log_trades=False):
    """回傳日報酬表；log_trades=True 時另回傳逐筆交易（報酬已含槓桿與來回成本）。"""
    C, O, ub, lb, sigma, lev, *_ = prep(es)
    checks = [c for c in C.columns if c.endswith((":00", ":30")) and "09:00" <= c <= "14:30"]
    out, log = [], []
    for day in C.index:
        if np.isnan(lev[day]) or np.isnan(sigma.loc[day, "09:00"]):
            continue
        k = lev[day] / O[day]
        pos, entry, t_in, ret, trades = 0, 0.0, None, 0.0, 0

        def close_trade(p, t_out):
            r = (pos * (p - entry) - 2 * COST_PTS) * k
            log.append((day, pos, t_in, t_out, r))
            return r

        for c in checks:
            p, u, l = C.at[day, c], ub.at[day, c], lb.at[day, c]
            new = pos
            if pos == 1 and p < u:
                new = 0
            elif pos == -1 and p > l:
                new = 0
            if new == 0:
                new = 1 if p > u else (-1 if p < l else 0)
            if new != pos:
                if pos != 0:
                    ret += close_trade(p, c)
                trades += abs(new - pos)
                pos, entry, t_in = new, p, c
        if pos != 0:
            ret += close_trade(C.at[day, "15:00"], "15:00")
        out.append((day, ret, trades, lev[day]))
    res = pd.DataFrame(out, columns=["day", "ret", "trades", "lev"]).set_index("day")
    if log_trades:
        return res, pd.DataFrame(log, columns=["day", "side", "t_in", "t_out", "ret"])
    return res


if __name__ == "__main__":
    es, vix = load("ES", ct=True), load("VIX", ct=True)
    res = run(es)
    vt = vix[vix.index.time <= pd.Timestamp("15:00").time()].Close
    vix_d = vt.groupby(vt.index.normalize()).last()
    res["vix_prev"] = vix_d.shift(1).reindex(res.index)      # 前一日收盤 VIX，無前視
    res["vix_pct"] = vix_d.rolling(252).rank(pct=True).shift(1).reindex(res.index)

    t = es.index.time
    esc = es[t <= pd.Timestamp("15:00").time()].Close
    bh = esc.groupby(esc.index.normalize()).last().pct_change().reindex(res.index)

    rows = [stats(res.ret, "日內動能（全部）"), stats(bh, "ES 買進持有")]
    hi = res.vix_pct > 0.5
    rows.append(stats(res.ret.where(hi, 0.0), "動能，只在 VIX 百分位 > 50% 交易"))
    rows.append(stats(res.ret.where(~hi, 0.0), "動能，只在 VIX 百分位 ≤ 50% 交易"))
    print(f"樣本 {res.index[0].date()} ~ {res.index[-1].date()}，{len(res)} 天\n")
    print(pd.DataFrame(rows).set_index("策略").to_string(formatters={"年化報酬": "{:.1%}".format, "年化波動": "{:.1%}".format, "Sharpe": "{:.2f}".format, "最大回撤": "{:.1%}".format, "交易日占比": "{:.0%}".format}))

    print("\n依前一日 VIX 水準分組：")
    g = res.groupby(pd.cut(res.vix_prev, [0, 15, 20, 30, 100]), observed=True).ret
    print(pd.DataFrame({"天數": g.size(), "日均報酬bp": g.mean() * 1e4,
                        "年化Sharpe": g.mean() / g.std() * np.sqrt(252)}).round(2).to_string())
    print("\n每年 Sharpe：", " ".join(f"{y}:{r.mean()/r.std()*np.sqrt(252):.2f}" for y, r in res.ret.groupby(res.index.year)))
    print(f"平均每日換手 {res.trades.mean():.2f} 次（單邊）")
    res.to_pickle("results/intraday_mom.pkl")
