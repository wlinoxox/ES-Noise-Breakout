"""日內動能策略在 2025 年之後失效的原因拆解。

1. 每年多單 / 空單的報酬與勝率
2. 依進場時間分組
3. 市場本身的日內特性：第一小時報酬 vs 當日其餘時段報酬的相關、30 分鐘報酬的一階自我相關
4. 雜訊區間寬度與突破頻率
進場時間同時列出芝加哥時間與台北時間（夏令/冬令）。
"""
import numpy as np
import pandas as pd
from load import load, tpe_clock
import intraday_mom as m

es = load("ES", ct=True)
res, log = m.run(es, log_trades=True)
log["year"] = log.day.dt.year
sr = lambda r: r.mean() / r.std() * np.sqrt(252)

print("== 126 日滾動 Sharpe（每季末取樣）")
roll = res.ret.rolling(126).apply(sr, raw=False)
print(roll.resample("QE").last().dropna().round(2).to_string())

print("\n== 每年多 / 空拆解（報酬為每年加總，%）")
g = log.groupby(["year", "side"]).ret
tab = pd.DataFrame({"筆數": g.size(), "總報酬%": g.sum() * 100, "勝率": g.apply(lambda x: (x > 0).mean()),
                    "平均bp": g.mean() * 1e4}).unstack("side")
print(tab.round(2).to_string())

print("\n== 依進場時間（2020–24 vs 2025–26，平均每筆 bp）")
log["era"] = np.where(log.year >= 2025, "2025-26", "2020-24")
log["bp"] = log.ret * 1e4
log["t_in"] = log.t_in + " CT（台北 " + log.t_in.map(tpe_clock) + "）"
print(log.pivot_table(index="t_in", columns="era", values="bp", aggfunc=["mean", "size"]).round(1).to_string())

print("\n== 出場方式：收盤出場 vs 中途被洗出")
log["exit"] = np.where(log.t_out == "15:00", "收盤", "停損/反手")
print(log.pivot_table(index="exit", columns="era", values="bp", aggfunc=["mean", "size"]).round(1).to_string())

print("\n== 市場日內特性（每年）")
t = es.index.time
rth = es[(t >= pd.Timestamp("08:30").time()) & (t <= pd.Timestamp("15:00").time())]
px = rth.Close.to_frame().assign(day=rth.index.normalize(), tod=rth.index.strftime("%H:%M"))
C = px.pivot(index="day", columns="tod", values="Close").ffill(axis=1)
C = C[C.notna().sum(axis=1) > 300]
O = rth.Open.groupby(rth.index.normalize()).first().reindex(C.index)
first = np.log(C["09:30"] / O)
rest = np.log(C["15:00"] / C["09:30"])
last30 = np.log(C["15:00"] / C["14:30"])
early = np.log(C["14:30"] / O)
half = C[[c for c in C.columns if c.endswith((":00", ":30"))]]
r30 = np.log(half).diff(axis=1).iloc[:, 1:]
ac = r30.groupby(r30.index.year).apply(lambda x: pd.Series(x.values[:, 1:].ravel()).corr(pd.Series(x.values[:, :-1].ravel())))
yr = pd.DataFrame({
    "corr(首小時, 其餘)": first.groupby(first.index.year).corr(rest),
    "corr(14:30前, 最後30分)": early.groupby(early.index.year).corr(last30),
    "30分報酬自相關": ac,
    "日內波動%": (np.log(C["15:00"] / O)).groupby(C.index.year).std() * np.sqrt(252) * 100,
    "動能策略Sharpe": res.ret.groupby(res.index.year).apply(sr),
})
print(yr.round(3).to_string())
