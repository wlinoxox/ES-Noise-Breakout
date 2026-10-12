"""事件研究：1 分 K 爆量紅 K（相對成交量高、收盤 > 開盤）之後，ES 的報酬是延續還是反轉？

相對量 = 該分鐘成交量 ÷ 過去 14 天同一分鐘的平均量（排除成交量的日內 U 型）。
只看正規盤 08:31–14:30 CT（台北 21:31–03:30 夏令 / 22:31–04:30 冬令）。
未來報酬從事件 K 棒收盤起算。

注意（2026-10-12 修正）：「日平均」是先把同一天的事件平均、再對所有天平均，等於每個事件的權重是
1 ÷ 當天事件總數，而事件總數要收盤才知道 → 有前視偏誤，不可當作可交易的報酬。
可交易的數字看「逐筆平均」與「每天第 1 個事件」。爆量黑 K 的日平均 +7.7bp 就是這個偏誤造成的；
逐筆平均約 −1.2bp、每天第 1 個事件約 −1.1bp（到收盤）。
"""
import numpy as np
import pandas as pd
from load import load

es = load("ES", ct=True)
t = es.index.time
rth = es[(t >= pd.Timestamp("08:30").time()) & (t <= pd.Timestamp("15:00").time())]
day, tod = rth.index.normalize(), rth.index.strftime("%H:%M")
piv = lambda s: s.to_frame("v").assign(day=day, tod=tod).pivot(index="day", columns="tod", values="v")
C, Op, V = piv(rth.Close).ffill(axis=1), piv(rth.Open), piv(rth.Vol.astype(float))
keep = C.notna().sum(axis=1) > 300
C, Op, V = C[keep], Op[keep], V[keep].fillna(0)
relv = V / V.rolling(14).mean().shift(1)                 # 只用過去 14 天
body = (C - Op) / C                                       # 紅 K：收盤 > 開盤
cols = [c for c in C.columns if "08:31" <= c <= "14:30"]
H = {"5分": 5, "15分": 15, "30分": 30, "60分": 60}
lc = np.log(C)
fwd = {h: (lc.shift(-n, axis=1) - lc)[cols] for h, n in H.items()}
fwd["到收盤"] = lc[["15:00"]].values - lc[cols]
nxt = lc["15:00"].shift(-1)                               # 隔天收盤
fwd["到隔天收盤"] = (nxt.values[:, None] - lc[cols])

R, B = relv[cols], body[cols]
def event_stats(mask, label):
    row = {"事件": label, "次數": int(mask.sum().sum()), "有事件的天數": int(mask.any(axis=1).sum())}
    first = mask & (mask.cumsum(axis=1) == 1)             # 每天第 1 個事件（可交易）
    for h, F in fwd.items():
        x = F.where(mask)
        dm = x.mean(axis=1).dropna()                      # 日平均（有前視偏誤，僅供對照）
        row[h + "｜日平均(偏誤)"] = dm.mean() * 1e4
        row[h + "｜逐筆"] = x.stack().dropna().mean() * 1e4
        f1 = F.where(first).stack().dropna()   # pandas 3 的 stack 不會自動丟掉 NaN
        row[h + "｜第1個"] = f1.mean() * 1e4
        row[h + "｜第1個 t"] = f1.mean() / f1.std() * np.sqrt(len(f1))
    return row

rows = [event_stats(R.notna() & B.notna(), "全部 K 棒（基準）")]
for th in [2, 3, 5]:
    rows.append(event_stats((R >= th) & (B > 0), f"相對量 ≥ {th}，紅 K"))
    rows.append(event_stats((R >= th) & (B < 0), f"相對量 ≥ {th}，黑 K"))
# 實體夠大的紅 K（收盤 − 開盤 ≥ 過去該分鐘平均 |報酬| 的 2 倍）
absr = (lc.diff(axis=1)).abs()
big = (C - Op) / C >= 2 * absr.rolling(14).mean().shift(1)
rows.append(event_stats((R >= 3) & big[cols], "相對量 ≥ 3，大紅 K"))
T = pd.DataFrame(rows).set_index("事件")
print(f"樣本 {C.index[0].date()} ~ {C.index[-1].date()}，正規盤 {len(cols)} 個分鐘點；報酬單位 bp\n")
for h in ["5分", "30分", "到收盤"]:
    print(f"── {h}")
    print(T[["次數"] + [h + s for s in ["｜日平均(偏誤)", "｜逐筆", "｜第1個", "｜第1個 t"]]].round(2).to_string())
print(f"\n參考：來回成本約 {0.5 / C['15:00'].median() * 1e4:.1f} bp（1 tick × 2）")
