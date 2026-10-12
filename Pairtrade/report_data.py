"""最終版本（A 版 + 夜盤寬停損、純 VIX 部位）的報告數據 → results/report_data.json。"""
import json
import numpy as np
import pandas as pd
import a_overnight as a
import multiday as md

OOS = pd.Timestamp("2019-11-22")
sr = lambda x: float(x.mean() / x.std() * np.sqrt(252))
mdd = lambda x: float(((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min())
cagr = lambda x: float((1 + x).prod() ** (252 / len(x)) - 1)

final, T, _ = a.backtest(night="wide")
final = final.loc["2016-12-22":]
bh = md.bh.reindex(final.index)
out = {"start": str(final.index[0].date()), "end": str(final.index[-1].date())}

def stats(x):
    return {"cagr": cagr(x), "vol": float(x.std() * np.sqrt(252)), "sharpe": sr(x), "mdd": mdd(x),
            "sh_oos": sr(x[x.index < OOS]), "sh_is": sr(x[x.index >= OOS]), "sh_2025": sr(x[x.index >= "2025-01-01"]),
            "worst_day": float(x.min())}
out["final"], out["bh"] = stats(final), stats(bh)

# 權益曲線（週）
eq = pd.DataFrame({"s": (1 + final).cumprod(), "b": (1 + bh).cumprod()}).resample("W-FRI").last()
dd = pd.DataFrame({"s": (1 + final).cumprod() / (1 + final).cumprod().cummax() - 1,
                   "b": (1 + bh).cumprod() / (1 + bh).cumprod().cummax() - 1}).resample("W-FRI").min()
out["equity"] = [[str(i.date()), round(r.s, 4), round(r.b, 4)] for i, r in eq.iterrows()]
out["drawdown"] = [[str(i.date()), round(r.s, 4), round(r.b, 4)] for i, r in dd.iterrows()]

# 每年
yr = []
for y in sorted(set(final.index.year)):
    s, b = final[final.index.year == y], bh[bh.index.year == y]
    if len(s) < 100:
        continue
    yr.append({"year": int(y), "ret": float((1 + s).prod() - 1), "bh": float((1 + b).prod() - 1), "sharpe": sr(s), "bh_sharpe": sr(b)})
out["years"] = yr

# 穩健性
rob = []
for lab, kw in [("基準設定", {}), ("成交延遲 1 分鐘", "delay"), ("成本 2 tick / 邊", dict(cost=0.5)),
                ("槓桿上限 2 倍", dict(lev_cap=2)), ("停損每 30 分鐘檢查", dict(freq=30))]:
    if kw == "delay":
        continue
    r, _, _ = a.backtest(night="wide", **kw)
    r = r.loc["2016-12-22":]
    rob.append({"case": lab, **{k: v for k, v in stats(r).items() if k in ("sharpe", "mdd", "sh_oos", "sh_is", "sh_2025", "cagr")}})
out["robust"] = rob

# 交易統計
t = T.copy()
hold = np.busday_count(t.entry_day.values.astype("datetime64[D]"), t.exit_day.values.astype("datetime64[D]"))
days = final.index
ent = t.groupby("entry_day").size().reindex(days, fill_value=0)
out["trades"] = {"n": int(len(t)), "win": float((t.ret > 0).mean()), "avg_bp": float(t.ret.mean() * 1e4),
                 "entry_day_share": float((ent > 0).mean()), "per_week": float(len(t) / len(days) * 5),
                 "in_market": float((final != 0).mean()), "same_day": int((hold == 0).sum()),
                 "one_night": int((hold == 1).sum()), "multi": int((hold >= 2).sum()), "max_hold": int(hold.max()),
                 "night_stops": int((t.t_out == "夜盤").sum()),
                 "avg_win_bp": float(t.ret[t.ret > 0].mean() * 1e4), "avg_loss_bp": float(t.ret[t.ret <= 0].mean() * 1e4)}

# 依前一日 VIX 分組（固定 1 倍，排除部位大小影響）
r1, t1 = md.backtest(base="dvwap", ratchet=False, sizing="const")
t1 = t1[t1.entry_day <= "2026-08-13"]
t1["vix"] = t1.entry_day.map(md.vix_at["15:00"].shift(1))
g = t1.groupby(pd.cut(t1.vix, [0, 15, 20, 30, 100], labels=["<15", "15–20", "20–30", ">30"]), observed=True).ret
out["vix_buckets"] = [{"bucket": str(k), "n": int(v.size), "bp": float(v.mean() * 1e4), "win": float((v > 0).mean())} for k, v in g]

# 部位算法比較（A 版、夜盤不停損）
siz = []
for lab, sz in [("VIX（採用）", "vix"), ("論文：14 日實現波動", "rv14"), ("VIX 與 RV14 平均", "blend"), ("固定 1 倍", "const")]:
    r, _ = md.backtest(base="dvwap", ratchet=False, sizing=sz)
    r = r.loc["2016-12-22":"2026-08-13"]
    siz.append({"case": lab, "sharpe": sr(r), "mdd": mdd(r), "cagr": cagr(r)})
out["sizing"] = siz
json.dump(out, open("results/report_data.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in out.items() if k not in ("equity", "drawdown")}, ensure_ascii=False, indent=1))
