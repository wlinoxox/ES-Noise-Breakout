"""資料與時區的完整檢查。每項印出 OK / 注意。"""
import numpy as np
import pandas as pd
from load import load, FILES, ROOT, _convert, CT, TPE

flag = lambda ok: "OK  " if ok else "注意"


def raw(name):
    """直接讀 CSV、不做任何時區轉換（用來對照）。"""
    df = pd.read_csv(ROOT / FILES[name], usecols=["Date", "Time", "Close"])
    tm = df.Time
    if tm.str.contains("午").any():
        tm = tm.str[3:] + np.where(tm.str.startswith("上午"), " AM", " PM")
        fmt = "%Y/%m/%d %I:%M:%S %p"
    else:
        fmt = "%Y/%m/%d %H:%M:%S"
    return pd.Series(df.Close.values, index=pd.to_datetime(df.Date + " " + tm, format=fmt), name=name)


print("== 1. 時區轉換")
for name in ["VIX", "NKD", "ES"]:
    r = raw(name)
    t = load(name)
    c = load(name, ct=True)
    dup_raw = r.index.duplicated().sum()
    print(f"[{name}] 原始 {len(r):,} 筆（原始重複時間 {dup_raw}），台北 {len(t):,}，芝加哥 {len(c):,}；捨棄 {len(r) - len(t)} 筆")
    if name == "ES":     # 原始為台北時間 → 台北版應與原始完全相同
        ok = t.Close.equals(r[~r.index.duplicated()].sort_index())
        print(f"  {flag(ok)} 台北版 = 原始 CSV")
    else:                # 原始為芝加哥時間 → 芝加哥版應與原始相同（扣掉秋季歧義那一小時）
        rr = r[~r.index.duplicated()].sort_index()
        common = rr.index.intersection(c.index)
        ok = (rr.loc[common] == c.Close.loc[common]).all() and len(rr) - len(common) < 200
        print(f"  {flag(ok)} 芝加哥版 = 原始 CSV（相同時間點價格一致；差 {len(rr) - len(common)} 筆）")
        lost = rr.index.difference(c.index)
        if len(lost):
            print(f"       捨棄的時間點都在秋季回撥日 01:00–01:59：{lost.hour.isin([1]).all()}，日期 {sorted(set(lost.date))[:4]} …")
    # 抽查：台北 = 芝加哥 + 13h（夏令）或 +14h（冬令）
    smp = c.index[:: max(1, len(c) // 2000)]
    off = (to := _convert(smp, CT, TPE)) - smp
    dst = smp.tz_localize(CT, ambiguous="NaT", nonexistent="NaT").map(lambda x: bool(x.dst()) if x is not pd.NaT else None)
    exp = np.where(dst, 13, 14)
    print(f"  {flag((off / pd.Timedelta('1h') == exp).all())} 抽查 {len(smp)} 點：夏令 +13h、冬令 +14h")

print("\n== 2. 交易時段在夏令 / 冬令都正確（芝加哥時間）")
es = load("ES", ct=True)
for lab, months in [("夏令(7月)", [7]), ("冬令(1月)", [1])]:
    x = es[es.index.month.isin(months)]
    h = x.groupby(x.index.hour).size()
    vm = x.Vol.groupby(x.index.strftime("%H:%M")).mean()
    print(f"  {lab}: 16 點筆數 {h.get(16, 0)}（維護時段應≈0）；成交量最大的分鐘 {vm.idxmax()}；"
          f"08:29 / 08:30 / 08:31 平均量 {vm['08:29']:.0f} / {vm['08:30']:.0f} / {vm['08:31']:.0f}")
esT = load("ES")
for lab, months in [("夏令(7月)", [7]), ("冬令(1月)", [1])]:
    x = esT[esT.index.month.isin(months)]
    vm = x.Vol.groupby(x.index.strftime("%H:%M")).mean()
    print(f"  台北時間 {lab}: 成交量最大的分鐘 {vm.idxmax()}（應為 {'21:30' if months == [7] else '22:30'} 附近）")
for d in ["2025-03-07", "2025-03-10", "2025-10-31", "2025-11-03"]:   # 美國換季前後
    x = esT.loc[d:pd.Timestamp(d) + pd.Timedelta("1D")]
    print(f"  台北 {d} 夜盤成交量最大分鐘：{x.Vol.idxmax()}")

print("\n== 3. 不同資料之間的時間對齊（1 分鐘報酬相關 vs 位移分鐘數）")
vix, nkd = load("VIX", ct=True), load("NKD", ct=True)
rth = lambda s: s[(s.index.time >= pd.Timestamp("08:31").time()) & (s.index.time <= pd.Timestamp("15:00").time())]
re_ = np.log(es.Close).diff()
for name, other in [("VIX", vix), ("NKD", nkd)]:
    ro = np.log(other.Close).diff()
    for lab, months in [("夏令", [6, 7, 8]), ("冬令", [12, 1, 2])]:
        res = {}
        for k in [-2, -1, 0, 1, 2]:
            x = pd.concat([re_, ro.shift(freq=f"{k}min")], axis=1, join="inner").dropna()
            x = rth(x[x.index.month.isin(months)])
            res[k] = x.corr().iloc[0, 1]
        best = max(res, key=lambda k: abs(res[k]))
        print(f"  {flag(best == 0)} ES vs {name} {lab}: " + " ".join(f"{k:+d}分:{v:+.3f}" for k, v in res.items()))

print("\n== 4. 資料品質（ES）")
e = load("ES")
bad_ohlc = ((e.High < e[["Open", "Close"]].max(axis=1)) | (e.Low > e[["Open", "Close"]].min(axis=1))).sum()
print(f"  {flag(bad_ohlc == 0)} OHLC 矛盾 {bad_ohlc} 筆；價格 ≤ 0：{(e[['Open','High','Low','Close']] <= 0).any(axis=1).sum()}；成交量 = 0：{(e.Vol == 0).sum()}")
ec = load("ES", ct=True)
gap = ec.index.to_series().diff()
r1 = np.log(ec.Close).diff()
big = r1[gap <= pd.Timedelta("2min")].abs().nlargest(8)
print("  最大 1 分鐘變動（連續兩分鐘內）：")
for t, v in big.items():
    print(f"     {t}（台北 {_convert(pd.DatetimeIndex([t]), CT, TPE)[0]}）{v:.2%}")
# 換月日（3/6/9/12 月第 3 個週五前 8 天左右）的跳空
d = ec.Close.groupby(ec.index.normalize()).last()
on = np.log(ec[ec.index.time == pd.Timestamp("17:00").time()].Open)
pc = np.log(ec[ec.index.time == pd.Timestamp("15:59").time()].Close)
pc.index = pc.index.normalize(); on.index = on.index.normalize()
g = (on - pc.reindex(on.index)).dropna()
print(f"  每日 15:59→17:00 跳空：中位數 {g.abs().median():.4%}，最大 5 個：", ", ".join(f"{i.date()} {v:+.2%}" for i, v in g.abs().nlargest(5).items()))

print("\n== 5. 日資料：沒有缺漏的交易日")
days = ec[(ec.index.time >= pd.Timestamp("08:30").time()) & (ec.index.time <= pd.Timestamp("15:00").time())]
n = days.groupby(days.index.normalize()).size()
bd = pd.bdate_range(n.index[0], n.index[-1])
miss = bd.difference(n.index)
print(f"  正規盤交易日 {len(n)}，平日中缺少 {len(miss)} 天（應多為美國假日）：", [str(x.date()) for x in miss[:12]], "…")
print(f"  正規盤分鐘數 < 300 的天數（提前收盤等）：{(n < 300).sum()}")
