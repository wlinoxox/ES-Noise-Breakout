"""依波動率調整 ES 部位（Moreira & Muir 2017 的 volatility-managed portfolio）。

部位在 t 日 15:00 CT（台北 04:00/05:00）依當時已知的資訊決定，賺 t→t+1 的報酬。
實現變異 RV 用 5 分鐘報酬（含夜盤）計算；VIX 取 15:00 CT 的值。
換倉成本：每單位部位變動 0.25 點（1 tick）。
"""
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant  # 不用 statsmodels.api：其 DLL 被 Windows 應用程式控制封鎖

COST_PTS, MAX_LEV, TARGET = 0.25, 2.0, 0.15

d = pd.read_pickle("results/daily.pkl")       # es_vix.py 產生
r = d.es.pct_change().shift(-1)               # t→t+1 報酬
rv21 = d.rv_d.rolling(21).sum() * 252 / 21    # 年化實現變異
rv5 = d.rv_d.rolling(5).sum() * 252 / 5
iv = (d.vix / 100) ** 2

# 權重（未縮放）；變異倒數型的 c 用全樣本把平均槓桿調成 1（只是尺度，不影響未設上限時的 Sharpe）
raw = {
    "1/RV21（月調整，MM 原版）": (1 / rv21).resample("ME").last().reindex(d.index, method="ffill"),
    "1/RV21（日調整）": 1 / rv21,
    "1/VIX²（日調整）": 1 / iv,
    "1/RV5（日調整，快）": 1 / rv5,
}
W = {"買進持有": pd.Series(1.0, index=d.index)}
for k, w in raw.items():
    W[k] = (w / w.mean()).clip(upper=MAX_LEV)
W["波動目標 15%（RV21）"] = (TARGET / np.sqrt(rv21)).clip(upper=MAX_LEV)
W["波動目標 15%（VIX）"] = (TARGET / (d.vix / 100)).clip(upper=MAX_LEV)
W["波動目標 15%（max(RV21,VIX)）"] = (TARGET / np.maximum(np.sqrt(rv21), d.vix / 100)).clip(upper=MAX_LEV)

start = rv21.first_valid_index()
rows, rets = [], {}
for k, w in W.items():
    w = w.loc[start:].ffill()
    cost = w.diff().abs().fillna(0) * COST_PTS / d.es.loc[start:]
    pr = (w * r.loc[start:] - cost).dropna()
    rets[k] = pr
    eq = (1 + pr).cumprod()
    yrs = len(pr) / 252
    rows.append({"策略": k, "年化報酬": eq.iloc[-1] ** (1 / yrs) - 1, "年化波動": pr.std() * np.sqrt(252),
                 "Sharpe": pr.mean() / pr.std() * np.sqrt(252), "最大回撤": (eq / eq.cummax() - 1).min(),
                 "平均槓桿": w.mean(), "年換手": w.diff().abs().sum() / yrs})
R = pd.DataFrame(rets)
bh = R["買進持有"]
for row in rows:
    y = R[row["策略"]]
    row["最差單日"] = y.min()
    if row["策略"] == "買進持有":     # 自己對自己回歸沒有意義
        row["年化alpha"] = row["alpha t"] = np.nan
        continue
    m = OLS(y, add_constant(bh), missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    row["年化alpha"] = m.params.iloc[0] * 252
    row["alpha t"] = m.tvalues.iloc[0]
fmt = {c: "{:.1%}".format for c in ["年化報酬", "年化波動", "最大回撤", "年化alpha", "最差單日"]}
fmt.update({c: "{:.2f}".format for c in ["Sharpe", "平均槓桿", "年換手", "alpha t"]})
print(f"樣本 {R.index[0].date()} ~ {R.index[-1].date()}，{len(R)} 天；槓桿上限 {MAX_LEV}\n")
print(pd.DataFrame(rows).set_index("策略").to_string(formatters=fmt))

print("\n每年 Sharpe：")
print(R.groupby(R.index.year).apply(lambda x: x.mean() / x.std() * np.sqrt(252)).T.round(2).to_string())
print("\n不含 2020 的 Sharpe：")
x = R[R.index.year != 2020]
print((x.mean() / x.std() * np.sqrt(252)).round(2).to_string())
R.to_pickle("results/vol_managed.pkl")
