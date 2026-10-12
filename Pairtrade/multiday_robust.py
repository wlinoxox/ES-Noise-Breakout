"""多日版兩個候選的穩健性：分段間隔、成本、延遲、槓桿上限。"""
import pandas as pd
from multiday import backtest, bh, summarize

cands = {"A 每日VWAP": dict(base="dvwap", ratchet=False), "B 進場起算VWAP＋分段": dict(base="avwap", ratchet=True)}
tests = [("基準", {}), ("成本 2 tick", dict(cost=0.5)), ("延遲 1 分", dict(delay=1)), ("槓桿上限 2", dict(lev_cap=2)),
         ("保守：延遲1分＋2tick＋槓桿≤2", dict(delay=1, cost=0.5, lev_cap=2))]
R, T = {"ES 買進持有": bh}, {}
for cn, ck in cands.items():
    for tn, tk in tests:
        R[f"{cn}｜{tn}"], T[f"{cn}｜{tn}"] = backtest(**{**ck, **tk})
for st in [0.5, 0.75, 1.5, 2.0]:
    k = f"B 進場起算VWAP＋分段｜間隔 ×{st}"
    R[k], T[k] = backtest(base="avwap", ratchet=True, step=st)
print(summarize(pd.DataFrame(R), T))
