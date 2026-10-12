"""ES vs NKD（日經 225 美元計價期貨）配對交易。

1. 共整合檢定（Engle-Granger）、價差半衰期
2. 滾動避險比率（只用過去資料）的價差 z 分數均值回歸
3. 依 VIX 狀態拆解
日資料取 15:00 CT（台北 04:00/05:00）收盤。成本：每單位價差部位變動 2bp（兩腳各約 1 tick）。
"""
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import coint
from load import load

COST = 2e-4


def daily_close(df):
    x = df[df.index.time <= pd.Timestamp("15:00").time()].Close
    return x.groupby(x.index.normalize()).last()


es, nkd, vix = (daily_close(load(n, ct=True)) for n in ["ES", "NKD", "VIX"])
p = pd.DataFrame({"es": np.log(es), "nkd": np.log(nkd), "vix": vix}).dropna()
print(f"樣本 {p.index[0].date()} ~ {p.index[-1].date()}，{len(p)} 天")
print(f"日報酬相關 {p.es.diff().corr(p.nkd.diff()):.3f}")

print("\n== 1. 共整合檢定（log NKD ~ log ES）")
for lab, x in [("全樣本", p)] + [(str(y), g) for y, g in p.groupby(p.index.year) if len(g) > 200]:
    t, pv, _ = coint(x.nkd, x.es)
    b = OLS(x.nkd, add_constant(x.es)).fit().params.es
    print(f"  {lab:>6}: EG t={t:6.2f}  p={pv:.3f}  β={b:.2f}")


def half_life(s):
    s = s.dropna()
    b = OLS(s.diff().iloc[1:], add_constant(s.shift(1).iloc[1:])).fit().params.iloc[1]
    return -np.log(2) / np.log(1 + b) if b < 0 else np.inf


def backtest(p, win, entry=2.0, exit=0.5, beta_mode="price"):
    """滾動 win 天回歸 log NKD = a + β·log ES；殘差 z 分數均值回歸。部位 +1 = 多 NKD、空 β 份 ES。"""
    if beta_mode == "price":
        cov = p.nkd.rolling(win).cov(p.es)
        beta = cov / p.es.rolling(win).var()
    else:                                    # 報酬避險比率
        rn, re = p.nkd.diff(), p.es.diff()
        beta = rn.rolling(win).cov(re) / re.rolling(win).var()
    alpha = p.nkd.rolling(win).mean() - beta * p.es.rolling(win).mean()
    spread = p.nkd - alpha - beta * p.es
    # z 分數用同一窗口殘差的標準差；beta/alpha 都只用到 t 日
    z = spread / spread.rolling(win).std()
    pos, cur = [], 0.0
    for zi in z.to_numpy():
        if np.isnan(zi):
            cur = 0.0
        elif cur == 0:
            cur = -1.0 if zi > entry else (1.0 if zi < -entry else 0.0)
        elif (cur == -1 and zi < exit) or (cur == 1 and zi > -exit):
            cur = 0.0
        pos.append(cur)
    pos = pd.Series(pos, index=p.index)
    # t 日收盤決定部位，賺 t→t+1；β 固定在進場當日
    ret = pos * (p.nkd.diff().shift(-1) - beta * p.es.diff().shift(-1)) - pos.diff().abs().fillna(0) * COST
    return ret.dropna(), pos, z, spread


def stats(r):
    eq = r.cumsum()
    return {"Sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan,
            "年化報酬": r.mean() * 252, "最大回撤": (eq - eq.cummax()).min(), "持倉比例": (r != 0).mean()}


print("\n== 2. 滾動避險的均值回歸（參數網格；報酬為單利、單位名目）")
rows = []
for mode in ["price", "return"]:
    for win in [20, 60, 120, 250]:
        for entry in [1.5, 2.0]:
            r, pos, z, spread = backtest(p, win, entry, beta_mode=mode)
            s = stats(r)
            s.update(避險=mode, 窗口=win, 進場z=entry, 半衰期=half_life(spread), 交易次數=int((pos.diff().abs() > 0).sum() / 2))
            rows.append(s)
G = pd.DataFrame(rows).set_index(["避險", "窗口", "進場z"])
print(G.to_string(formatters={"Sharpe": "{:.2f}".format, "年化報酬": "{:.1%}".format, "最大回撤": "{:.1%}".format,
                              "持倉比例": "{:.0%}".format, "半衰期": "{:.1f}".format}))

print("\n== 3. 依 VIX 狀態（前一日 VIX 的 252 日百分位），以 price/60/2.0 為例")
r, pos, z, _ = backtest(p, 60, 2.0)
vp = p.vix.rolling(252).rank(pct=True).shift(1).reindex(r.index)
for lab, msk in [("VIX 百分位 ≤ 50%", vp <= 0.5), ("VIX 百分位 > 50%", vp > 0.5), ("VIX > 25", p.vix.shift(1).reindex(r.index) > 25)]:
    x = r[msk & (pos.reindex(r.index) != 0)]
    print(f"  {lab:16s}: 持倉天數 {len(x):4d}  日均 {x.mean()*1e4:6.2f}bp  勝率 {(x>0).mean():.0%}")
r.to_pickle("results/es_nkd_pair.pkl")
