# trading

## ES 雜訊區間突破策略（`Pairtrade/`）

以 Zarattini, Aziz & Barbon (2024) 的日內動能策略為基礎，套用到 ES 期貨：只做多、跌破 VWAP 停損、可留倉過夜、以 VIX 決定部位。
2016-12 ~ 2026-08 回測 Sharpe 1.13、最大回撤 −8.4%（ES 買進持有 0.72 / −34.0%）。

- 最終版本與完整研究紀錄：[`Pairtrade/final/README.md`](Pairtrade/final/README.md)
- 研究過程的腳本：`Pairtrade/*.py`

### 資料（未包含在 repo）

行情資料檔案太大且有授權限制，沒有上傳。執行前請把以下檔案放在 `Pairtrade/`：

| 檔案 | 內容 |
|---|---|
| `ES #F Chart 2025-09-23-10-05.csv` | ES 連續合約 1 分 K（含成交量），台北時間 |
| `$VIX 1m.csv` | VIX 1 分 K，芝加哥時間 |
| `NKD #F_1min.csv` | NKD 連續合約 1 分 K，芝加哥時間（只有研究腳本用到） |

執行：

```bash
cd Pairtrade/final
PYTHONIOENCODING=utf-8 python run_backtest.py
```

需要 pandas、numpy、statsmodels（研究腳本另需 pypdf 讀論文）。
