"""重新產生 mentor 報告：計算數據 → report/report_data.json → 套入 report/template.html → report/es_strategy_report.html。

用法：cd Pairtrade/final && PYTHONIOENCODING=utf-8 python build_report.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import strategy as s

START = "2016-12-22"
HERE = Path(__file__).resolve().parent / "report"


def main():
    final, T = s.backtest()
    final = final.loc[START:]
    bh = s.buy_hold.reindex(final.index)
    out = {"start": str(final.index[0].date()), "end": str(final.index[-1].date()),
           "final": s.stats(final), "bh": s.stats(bh)}

    cum = lambda r: (1 + r).cumprod()
    eq = pd.DataFrame({"s": cum(final), "b": cum(bh)}).resample("W-FRI").last()
    dd = pd.DataFrame({"s": cum(final) / cum(final).cummax() - 1, "b": cum(bh) / cum(bh).cummax() - 1}).resample("W-FRI").min()
    out["equity"] = [[str(i.date()), round(r.s, 4), round(r.b, 4)] for i, r in eq.iterrows()]
    out["drawdown"] = [[str(i.date()), round(r.s, 4), round(r.b, 4)] for i, r in dd.iterrows()]

    sr = lambda x: float(x.mean() / x.std() * np.sqrt(252))
    out["years"] = [{"year": int(y), "ret": float((1 + final[final.index.year == y]).prod() - 1),
                     "bh": float((1 + bh[bh.index.year == y]).prod() - 1),
                     "sharpe": sr(final[final.index.year == y]), "bh_sharpe": sr(bh[bh.index.year == y])}
                    for y in sorted(set(final.index.year)) if (final.index.year == y).sum() >= 100]

    keys = ("cagr", "sharpe", "mdd", "sh_oos", "sh_is", "sh_2025")
    out["robust"] = []
    for lab, kw in [("基準設定", {}), ("成本 2 tick / 邊", dict(cost=0.5)), ("槓桿上限 2 倍", dict(lev_cap=2)),
                    ("停損每 30 分鐘檢查", dict(freq=30))]:
        x = s.stats(s.backtest(**kw)[0].loc[START:])
        out["robust"].append({"case": lab, **{k: x[k] for k in keys}})

    t = T[T.entry_day >= START]
    hold = np.busday_count(t.entry_day.values.astype("datetime64[D]"), t.exit_day.values.astype("datetime64[D]"))
    ent = t.groupby("entry_day").size().reindex(final.index, fill_value=0)
    out["trades"] = {"n": int(len(t)), "win": float((t.ret > 0).mean()), "avg_bp": float(t.ret.mean() * 1e4),
                     "entry_day_share": float((ent > 0).mean()), "per_week": float(len(t) / len(final) * 5),
                     "in_market": float((final != 0).mean()), "same_day": int((hold == 0).sum()),
                     "one_night": int((hold == 1).sum()), "multi": int((hold >= 2).sum()), "max_hold": int(hold.max()),
                     "night_stops": int((t.t_out == "夜盤").sum()),
                     "avg_win_bp": float(t.ret[t.ret > 0].mean() * 1e4), "avg_loss_bp": float(t.ret[t.ret <= 0].mean() * 1e4)}

    # 依前一天 VIX 分組：固定 1 倍、夜盤不停損，排除部位大小的影響
    _, t1 = s.backtest(night=None, sizing="const")
    t1 = t1.assign(vix=t1.entry_day.map(s.VIX["15:00"].shift(1)))
    g = t1.groupby(pd.cut(t1.vix, [0, 15, 20, 30, 100], labels=["<15", "15–20", "20–30", ">30"]), observed=True).ret
    out["vix_buckets"] = [{"bucket": str(k), "n": int(v.size), "bp": float(v.mean() * 1e4), "win": float((v > 0).mean())} for k, v in g]

    out["sizing"] = []
    for lab, sz in [("VIX（採用）", "vix"), ("論文：14 日實現波動", "rv14"), ("VIX 與 RV14 平均", "blend"), ("固定 1 倍", "const")]:
        x = s.stats(s.backtest(night=None, sizing=sz)[0].loc[START:])
        out["sizing"].append({"case": lab, "sharpe": x["sharpe"], "mdd": x["mdd"], "cagr": x["cagr"]})

    js = json.dumps(out, ensure_ascii=False, separators=(",", ":"))
    (HERE / "report_data.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    html = (HERE / "template.html").read_text(encoding="utf-8").replace("__DATA__", js.replace("</", "<\\/"))
    (HERE / "es_strategy_report.html").write_text(html, encoding="utf-8")
    print(f"已輸出 {HERE / 'es_strategy_report.html'}：Sharpe {out['final']['sharpe']:.2f}，"
          f"最大回撤 {out['final']['mdd']:.1%}，{out['trades']['n']} 筆交易")


if __name__ == "__main__":
    main()
