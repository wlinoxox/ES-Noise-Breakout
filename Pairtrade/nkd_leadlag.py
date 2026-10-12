"""ES 是否在分鐘級別領先 NKD？依交易時段拆解，並用 NKD 真實成交價估算可交易性。

時段以芝加哥時間切分：亞洲 18:00–01:59 CT（東京盤，台北約 07:00–15:00）、歐洲 02:00–08:29 CT、
美國正規盤 08:30–15:00 CT（台北 21:30–04:00 夏令 / 22:30–05:00 冬令）。
NKD 流動性低（約一半的分鐘沒有成交），所以：
- 相關分析用 5 分鐘格點的最後成交價（會因 NKD 價格停滯而高估領先性）
- 交易測試以「訊號後下一筆 NKD 實際成交」進場、5 分鐘後的下一筆成交出場
"""
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from load import load

TICK = 5.0   # NKD 一檔 = 5 點


def session(idx):
    h = idx.hour + idx.minute / 60
    return np.select([(h >= 18) | (h < 2), (h >= 2) & (h < 8.5), (h >= 8.5) & (h <= 15)],
                     ["亞洲", "歐洲", "美國"], "其他")


es, nkd = load("ES", ct=True), load("NKD", ct=True)
lp = pd.DataFrame({"es": np.log(es.Close), "nkd": np.log(nkd.Close)})
g5 = lp.resample("5min", label="right", closed="right").last()
cnt = nkd.Close.resample("5min", label="right", closed="right").count()
r = g5.diff()
r = r[(cnt > 0) & (cnt.shift(1) > 0)].dropna()     # 只用 NKD 前後都有成交的區間
r = r[r.index.to_series().diff().shift(-1) == pd.Timedelta("5min")]  # 下一格必須緊接著
nxt = g5.diff().shift(-1).reindex(r.index)
r["es_next"], r["nkd_next"] = nxt.es, nxt.nkd
r["sess"] = session(r.index)

print("== 1. 5 分鐘領先落後（依時段）")
for s, x in r.groupby("sess"):
    if s == "其他":
        continue
    print(f"  {s}: n={len(x):6d}  同期 corr={x.es.corr(x.nkd):.3f}  "
          f"corr(ES_t, NKD_t+1)={x.es.corr(x.nkd_next):+.3f}  corr(NKD_t, ES_t+1)={x.nkd.corr(x.es_next):+.3f}")

print("\n== 2. 預測迴歸 NKD_t+1 = a + b·ES_t + c·NKD_t（bp），依年份、亞洲時段")
a = r[r.sess == "亞洲"]
for y, x in a.groupby(a.index.year):
    mdl = OLS(x.nkd_next * 1e4, add_constant(x[["es", "nkd"]] * 1e4)).fit(cov_type="HC1")
    print(f"  {y}: b={mdl.params.es:+.3f} (t={mdl.tvalues.es:5.1f})  c={mdl.params.nkd:+.3f} (t={mdl.tvalues.nkd:5.1f})  R²={mdl.rsquared:.3f}")

print("\n== 3. 可交易性：以 NKD 實際成交價執行")
# 訊號：過去 5 分鐘 ES 報酬 − β·NKD 報酬（NKD 尚未追上的部分）；β 用前一年估計
nk = nkd.Close
trade_t = nk.index.values
r["beta"] = r.groupby(r.index.year).apply(
    lambda x: x.es.cov(x.nkd) / x.es.var()).shift(1).reindex(r.index.year).values
r["gap"] = r.beta * r.es - r.nkd
x = r.dropna(subset=["gap"]).copy()
# 進場 = 訊號時點之後第一筆 NKD 成交（至少延遲 1 分鐘）；出場 = 進場後 5 分鐘之後第一筆成交
i_in = np.searchsorted(trade_t, (x.index + pd.Timedelta("1min")).values)
ok = i_in < len(nk)
x, i_in = x[ok], i_in[ok]
t_in = trade_t[i_in]
i_out = np.searchsorted(trade_t, t_in + np.timedelta64(5, "m"))
ok = (i_out < len(nk)) & ((t_in - x.index.values) <= np.timedelta64(3, "m"))   # 3 分鐘內要成交得到
x, i_in, i_out = x[ok], i_in[ok], i_out[ok]
px_in, px_out = nk.values[i_in], nk.values[i_out]
x["fwd_bp"] = np.log(px_out / px_in) * 1e4
x["cost_bp"] = 2 * TICK / px_in * 1e4           # 來回各吃 1 檔（跨越買賣價差）
x["q"] = pd.qcut(x.gap, 10, labels=False)
tab = x.groupby(["sess", "q"]).agg(n=("fwd_bp", "size"), gap_bp=("gap", lambda v: v.mean() * 1e4),
                                   fwd_bp=("fwd_bp", "mean")).unstack("sess")
print("訊號十分位 → 之後 5 分鐘 NKD 報酬（bp，可成交價）")
print(tab.xs("fwd_bp", axis=1, level=0).round(2).to_string())
print(f"來回成本約 {x.cost_bp.median():.2f} bp")

print("\n極端訊號（最高 / 最低十分位）順勢交易的每筆淨報酬（bp）：")
x["side"] = np.select([x.q == 9, x.q == 0], [1, -1], 0)
t = x[x.side != 0]
t = t.assign(net=t.side * t.fwd_bp - t.cost_bp)
print(t.groupby(["sess", t.index.year]).net.agg(["size", "mean"]).unstack("sess").round(2).to_string())
