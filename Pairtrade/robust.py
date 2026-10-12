"""VWAP 停損·5 分·可留倉 的穩健性測試：成交延遲、成本、槓桿上限、檢查頻率。"""
import numpy as np
import pandas as pd
import intraday_mom as m
from mom_vwap import backtest, bh

OOS = pd.Timestamp("2019-11-22")   # 之前的資料是後來才拿到的 → 樣本外
base = dict(stop="vwap", freq=5, carry=True)
cases = [
    ("基準（同一根 K 棒成交、1 tick、槓桿≤4）", {}),
    ("成交延遲 1 分鐘", dict(delay=1)),
    ("成交延遲 2 分鐘", dict(delay=2)),
    ("成本 2 tick", dict(cost=0.5)),
    ("成本 4 tick", dict(cost=1.0)),
    ("槓桿上限 2", dict(lev_cap=2)),
    ("槓桿上限 1", dict(lev_cap=1)),
    ("每 1 分鐘檢查", dict(freq=1)),
    ("每 15 分鐘檢查", dict(freq=15)),
    ("保守組合：延遲 1 分＋2 tick＋槓桿≤2", dict(delay=1, cost=0.5, lev_cap=2)),
]
sr = lambda x: x.mean() / x.std() * np.sqrt(252)
mdd = lambda x: ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min()
cagr = lambda x: (1 + x).prod() ** (252 / len(x)) - 1
rows = []
for lab, kw in [("ES 買進持有", None)] + cases:
    r = bh.dropna() if kw is None else backtest(**{**base, **kw})[0]
    r = r.loc["2016-12-22":]
    rows.append({"情境": lab, "年化報酬": cagr(r), "Sharpe": sr(r), "最大回撤": mdd(r),
                 "Sharpe 樣本外": sr(r[r.index < OOS]), "Sharpe 2019-11後": sr(r[r.index >= OOS]),
                 "Sharpe 2025後": sr(r[r.index >= "2025-01-01"])})
f2, pc = "{:.2f}".format, "{:.1%}".format
print(pd.DataFrame(rows).set_index("情境").to_string(formatters={
    "年化報酬": pc, "最大回撤": pc, "Sharpe": f2, "Sharpe 樣本外": f2, "Sharpe 2019-11後": f2, "Sharpe 2025後": f2}))
