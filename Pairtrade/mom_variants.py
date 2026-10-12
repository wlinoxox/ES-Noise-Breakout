"""日內動能策略的改良版本：只做多、用 VIX 決定槓桿。

原版槓桿 = min(4, 2% / 過去 14 日報酬標準差)。
VIX 版槓桿 = min(4, 2% / (前一日 VIX/100/√252))，即以 VIX 隱含的日波動做目標。
"""
import numpy as np
import pandas as pd
from load import load
import intraday_mom as m

es, vix = load("ES", ct=True), load("VIX", ct=True)
res, log = m.run(es, log_trades=True)
vt = vix[vix.index.time <= pd.Timestamp("15:00").time()].Close
vix_prev = vt.groupby(vt.index.normalize()).last().shift(1).reindex(res.index)
lev_vix = (m.TARGET_VOL / (vix_prev / 100 / np.sqrt(252))).clip(upper=m.MAX_LEV)

log["unit"] = log.ret / log.day.map(res.lev)          # 槓桿 = 1 時的報酬
daily = lambda l: l.groupby("day").unit.sum().reindex(res.index).fillna(0)
both, longs = daily(log), daily(log[log.side == 1])

variants = {
    "雙向 × 原版槓桿": both * res.lev,
    "雙向 × VIX 槓桿": both * lev_vix,
    "只做多 × 原版槓桿": longs * res.lev,
    "只做多 × VIX 槓桿": longs * lev_vix,
}
R = pd.DataFrame(variants).dropna()
rows = [m.stats(R[k], k) for k in R]
print(f"樣本 {R.index[0].date()} ~ {R.index[-1].date()}\n")
print(pd.DataFrame(rows).set_index("策略").to_string(formatters={
    "年化報酬": "{:.1%}".format, "年化波動": "{:.1%}".format, "Sharpe": "{:.2f}".format,
    "最大回撤": "{:.1%}".format, "交易日占比": "{:.0%}".format}))
print("\n每年 Sharpe：")
print(R.groupby(R.index.year).apply(lambda x: x.mean() / x.std() * np.sqrt(252)).T.round(2).to_string())
print(f"\n平均槓桿：原版 {res.lev.mean():.2f}，VIX 版 {lev_vix.mean():.2f}；兩者相關 {res.lev.corr(lev_vix):.2f}")
