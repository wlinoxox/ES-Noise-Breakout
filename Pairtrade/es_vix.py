"""ES 與 VIX 的關係：同期相關、不對稱性、領先落後、VIX 對未來報酬的預測力、波動風險溢酬。

資料以台北時間儲存；交易時段用芝加哥時間 (CT) 判斷：正規盤 08:30–15:00 CT
（台北 21:30–04:00 美國夏令 / 22:30–05:00 冬令）。日資料取 15:00 CT（台北 04:00/05:00）收盤。
"""
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant  # 不用 statsmodels.api：其 DLL 被 Windows 應用程式控制封鎖
from load import load

ANN = np.sqrt(252)


def nw(y, X, lags=5):
    """Newey-West OLS，回傳 (params, tvalues, r2)。"""
    m = OLS(y, add_constant(X), missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return m.params, m.tvalues, m.rsquared


def rth(df):
    t = df.index.time
    return df[(t >= pd.Timestamp("08:30").time()) & (t <= pd.Timestamp("15:00").time())]


es, vix = load("ES", ct=True), load("VIX", ct=True)
es_r, vix_r = rth(es), rth(vix)

# ---------- 日資料 ----------
d = pd.DataFrame({
    "es": es_r.Close.groupby(es_r.index.date).last(),
    "es_open": es_r.Open.groupby(es_r.index.date).first(),
    "vix": vix_r.Close.groupby(vix_r.index.date).last(),
}).dropna()
d.index = pd.to_datetime(d.index)
d["ret"] = np.log(d.es).diff()
d["dvix"] = d.vix.diff()
d["dlvix"] = np.log(d.vix).diff()
print(f"日資料 {d.index[0].date()} ~ {d.index[-1].date()}, {len(d)} 天\n")

print("== 1. 同期關係（日）")
print(f"corr(ES 報酬, ΔVIX)       = {d.ret.corr(d.dvix):.3f}")
print(f"corr(ES 報酬, Δlog VIX)   = {d.ret.corr(d.dlvix):.3f}")
neg = d.ret.clip(upper=0)
p, t, r2 = nw(d.dvix, pd.DataFrame({"ret": d.ret * 100, "ret_neg": neg * 100}))
print(f"ΔVIX = a + b·ret% + c·min(ret%,0):  b={p.ret:.2f} (t={t.ret:.1f})  c={p.ret_neg:.2f} (t={t.ret_neg:.1f})  R²={r2:.2f}")
print(f"  → ES 跌 1% 時 VIX ≈ {-(p.ret + p.ret_neg):+.2f} 點；漲 1% 時 ≈ {p.ret:+.2f} 點")
# 依 VIX 水準分組的 beta
d["vix_lvl"] = pd.cut(d.vix.shift(1), [0, 15, 20, 30, 100], labels=["<15", "15-20", "20-30", ">30"])
print("  依前一日 VIX 水準分組：ΔVIX 對 ES 報酬% 的斜率，與 Δlog VIX 的相關")
for k, g in d.groupby("vix_lvl", observed=True):
    b = np.polyfit(g.ret.dropna() * 100, g.dvix[g.ret.notna()], 1)[0]
    print(f"    VIX {k:>6}: n={len(g):4d}  斜率={b:6.2f}  corr={g.ret.corr(g.dlvix):.2f}")
roll = d.ret.rolling(63).corr(d.dlvix)
print(f"  63 日滾動相關：中位數 {roll.median():.2f}，最高 {roll.max():.2f}（{roll.idxmax().date()}），最低 {roll.min():.2f}")
# 同漲同跌（spot-up/vol-up）的比例
both_up = ((d.ret > 0) & (d.dvix > 0)).mean()
print(f"  ES 與 VIX 同一天一起上漲的比例：{both_up:.1%}")
yr = d.groupby(d.index.year).apply(lambda g: g.ret.corr(g.dlvix))
print("  每年的相關：", " ".join(f"{y}:{c:.2f}" for y, c in yr.items()))

# ---------- 日內領先落後 ----------
print("\n== 2. 日內領先落後（5 分鐘，正規盤）")
m = pd.DataFrame({"es": np.log(es_r.Close), "vix": np.log(vix_r.Close)}).dropna()
m5 = m.resample("5min", label="right", closed="right").last().dropna()
m5 = rth(m5)
g = m5.groupby(m5.index.date)
r5 = g.diff().dropna()   # 只算同一天內的報酬
print(f"同期 corr = {r5.es.corr(r5.vix):.3f}")
for k in [1, 2, 3, 6]:
    a = r5.es.corr(r5.vix.groupby(r5.index.date).shift(-k))   # ES 領先 VIX
    b = r5.vix.corr(r5.es.groupby(r5.index.date).shift(-k))   # VIX 領先 ES
    print(f"  落後 {k*5:2d} 分鐘：corr(ES_t, VIX_t+k)={a:+.3f}   corr(VIX_t, ES_t+k)={b:+.3f}")

# ---------- VIX 對未來 ES 報酬的預測力 ----------
print("\n== 3. VIX 能否預測未來 ES 報酬？")
for h in [1, 5, 21]:
    d[f"fwd{h}"] = np.log(d.es).shift(-h) - np.log(d.es)
d["vix_pct"] = d.vix.rolling(252).rank(pct=True)
d["q"] = pd.qcut(d.vix, 5, labels=["Q1 低", "Q2", "Q3", "Q4", "Q5 高"])
tab = d.groupby("q", observed=True).agg(vix_mean=("vix", "mean"), n=("vix", "size"),
    fwd5=("fwd5", "mean"), fwd21=("fwd21", "mean"), hit21=("fwd21", lambda x: (x > 0).mean()),
    sd21=("fwd21", "std"))
tab[["fwd5", "fwd21", "sd21"]] *= 100
print("依 VIX 水準五分位（全樣本切點，有前視，僅供描述）：")
print(tab.round(2).to_string())
for h in [5, 21]:
    p, t, r2 = nw(d[f"fwd{h}"] * 100, d[["vix"]], lags=h)
    print(f"  fwd{h}% = a + b·VIX：b={p.vix:.3f} (NW t={t.vix:.2f}) R²={r2:.3f}")
# VIX 突刺：高於 20 日均線 25%
d["spike"] = d.vix / d.vix.rolling(20).mean() > 1.25
first = d.spike & ~d.spike.shift(1, fill_value=False)
s = d[first]
print(f"  VIX 突刺事件（VIX > 1.25×MA20，首日）n={len(s)}：未來 5 日 ES 平均 {s.fwd5.mean()*100:.2f}%，"
      f"21 日 {s.fwd21.mean()*100:.2f}%（勝率 {(s.fwd21>0).mean():.0%}）；全樣本 21 日 {d.fwd21.mean()*100:.2f}%")

# ---------- 波動風險溢酬 VRP ----------
print("\n== 4. 波動風險溢酬（VIX vs 未來 21 日實現波動）")
e5 = np.log(es.Close).resample("5min").last().dropna()
r = e5.diff().dropna()
rv_d = (r**2).groupby(r.index.date).sum()   # 含夜盤的 24h 實現變異
rv_d.index = pd.to_datetime(rv_d.index)
d["rv_d"] = rv_d.reindex(d.index)
d["rv_fwd21"] = np.sqrt(d.rv_d[::-1].rolling(21).sum()[::-1].shift(-1) * 252 / 21) * 100
d["rv_past21"] = np.sqrt(d.rv_d.rolling(21).sum() * 252 / 21) * 100
d["vrp"] = d.vix - d.rv_fwd21
print(f"VIX 平均 {d.vix.mean():.1f}，未來 21 日實現波動平均 {d.rv_fwd21.mean():.1f}，"
      f"VRP 平均 {d.vrp.mean():.1f} 點；VIX > 實現波動的比例 {(d.vrp>0).mean():.0%}")
d["vrp_ex_ante"] = d.vix - d.rv_past21   # 只用已知資料
p, t, r2 = nw(d.fwd21 * 100, d[["vrp_ex_ante"]], lags=21)
print(f"  fwd21% = a + b·(VIX − 過去 21 日 RV)：b={p.vrp_ex_ante:.3f} (NW t={t.vrp_ex_ante:.2f}) R²={r2:.3f}")

d.to_pickle("results/daily.pkl")
