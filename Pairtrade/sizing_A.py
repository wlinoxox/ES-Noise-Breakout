"""A 版（每日 VWAP 停損、多日持有）的部位算法比較，以及依進場時 VIX 分組的表現。"""
import numpy as np
import pandas as pd
import multiday as md

cases = [("VIX（目前）", "vix"), ("論文 RV14", "rv14"), ("VIX＋RV14 平均", "blend"), ("固定 1 倍", "const")]
R, T = {"ES 買進持有": md.bh}, {}
for lab, sz in cases:
    R[lab], T[lab] = md.backtest(base="dvwap", ratchet=False, sizing=sz)
    k = f"{lab}｜2 tick"
    R[k], T[k] = md.backtest(base="dvwap", ratchet=False, sizing=sz, cost=0.5)
print(md.summarize(pd.DataFrame(R), T))

# 依進場時 VIX 分組：用固定 1 倍版本看「每筆交易的原始報酬」，排除部位大小的影響
t = T["固定 1 倍"].copy()
ent = md.vix_at.stack()
# 進場時點的 VIX：用進場日的前一日收盤近似分組（逐筆紀錄未存進場分鐘）
vprev = md.vix_at["15:00"].shift(1)
t["vix"] = t.entry_day.map(vprev)
t["組"] = pd.cut(t.vix, [0, 15, 20, 30, 100], labels=["<15", "15–20", "20–30", ">30"])
g = t.groupby("組", observed=True).ret
print("\n依前一日 VIX 分組（固定 1 倍、每筆報酬 bp）：")
print(pd.DataFrame({"筆數": g.size(), "平均bp": g.mean() * 1e4, "勝率": g.apply(lambda x: (x > 0).mean()),
                    "每筆 Sharpe×√筆數": g.mean() / g.std() * np.sqrt(g.size())}).round(2).to_string())
