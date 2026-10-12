"""執行最終版本回測，印出主要結果、每年表現與穩健性測試。

用法：cd Pairtrade/final && PYTHONIOENCODING=utf-8 python run_backtest.py
"""
import numpy as np
import pandas as pd
import strategy as s

START = "2016-12-22"


def row(label, r):
    x = s.stats(r.loc[START:])
    return {"情境": label, "年化報酬": x["cagr"], "年化波動": x["vol"], "Sharpe": x["sharpe"], "最大回撤": x["mdd"],
            "2016-12~2019-11": x["sh_oos"], "2019-11後": x["sh_is"], "2025後": x["sh_2025"]}


fmt = {c: "{:.1%}".format for c in ["年化報酬", "年化波動", "最大回撤"]}
fmt.update({c: "{:.2f}".format for c in ["Sharpe", "2016-12~2019-11", "2019-11後", "2025後"]})

if __name__ == "__main__":
    final, trades = s.backtest()
    print(f"樣本 {START} ~ {final.index[-1].date()}（2016-12-22 起，前 14 天用來計算雜訊區間）\n")

    print("== 最終版本 vs ES 買進持有")
    print(pd.DataFrame([row("最終版本", final), row("ES 買進持有", s.buy_hold)]).set_index("情境").to_string(formatters=fmt))

    print("\n== 穩健性")
    cases = [("成本 2 tick / 邊", dict(cost=0.5)), ("槓桿上限 2 倍", dict(lev_cap=2)),
             ("停損每 30 分鐘檢查", dict(freq=30)), ("夜盤不停損", dict(night=None)),
             ("夜盤跌破前一天 VWAP 即停損", dict(night="vwap"))]
    print(pd.DataFrame([row(lab, s.backtest(**kw)[0]) for lab, kw in cases]).set_index("情境").to_string(formatters=fmt))

    print("\n== 部位算法（夜盤不停損）")
    print(pd.DataFrame([row(lab, s.backtest(night=None, sizing=sz)[0]) for lab, sz in
                        [("VIX（採用）", "vix"), ("論文 14 日實現波動", "rv14"), ("VIX 與 RV14 平均", "blend"),
                         ("固定 1 倍", "const")]]).set_index("情境").to_string(formatters=fmt))

    t = trades[trades.entry_day >= START]
    hold = np.busday_count(t.entry_day.values.astype("datetime64[D]"), t.exit_day.values.astype("datetime64[D]"))
    print(f"\n== 交易：{len(t)} 筆，勝率 {(t.ret > 0).mean():.0%}，平均 {t.ret.mean()*1e4:.1f} bp；"
          f"當天出場 {(hold == 0).sum()}、留倉 1 晚 {(hold == 1).sum()}、2 晚以上 {(hold >= 2).sum()}；"
          f"夜盤停損 {(t.t_out == '夜盤').sum()} 次")

    r = final.loc[START:]
    print("\n== 每年 Sharpe")
    print(r.groupby(r.index.year).apply(lambda x: x.mean() / x.std() * np.sqrt(252)).round(2).to_string())
