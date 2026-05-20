from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_lab.public_data import _download_fred_series, download_yahoo_chart, write_public_data  # noqa: E402


FRED_ANNUAL_SERIES = {
    "fed_funds": "DFF",
    "fed_funds_monthly": "FEDFUNDS",
    "treasury_3m": "TB3MS",
    "yield_2y": "DGS2",
    "yield_5y": "DGS5",
    "yield_10y": "DGS10",
    "yield_30y": "DGS30",
    "curve_10y_2y": "T10Y2Y",
    "hy_oas": "BAMLH0A0HYM2",
    "ig_oas": "BAMLC0A0CM",
    "baa": "BAA",
    "aaa": "AAA",
    "excess_bond_premium": "EBP",
    "unemployment": "UNRATE",
    "cpi": "CPIAUCSL",
    "core_cpi": "CPILFESL",
    "pce": "PCEPI",
    "core_pce": "PCEPILFE",
    "breakeven_5y": "T5YIE",
    "breakeven_10y": "T10YIE",
    "gasoline_yoy_source": "GASREGW",
    "industrial_production": "INDPRO",
    "real_gdp": "GDPC1",
    "ism_manufacturing": "NAPM",
    "retail_sales_yoy_source": "RSAFS",
    "consumer_confidence": "UMCSENT",
    "initial_claims": "ICSA",
    "payrolls": "PAYEMS",
    "lei": "USSLIND",
    "recession_dummy": "USREC",
    "money_market_assets": "MMMFFAQ027S",
    "financial_conditions_index": "NFCI",
    "bank_lending_standards": "DRTSCILM",
    "fed_balance_sheet": "WALCL",
    "financial_stress": "STLFSI4",
    "m2": "M2SL",
    "commercial_bank_credit": "TOTBKCR",
    "reverse_repo_level": "RRPONTSYD",
    "treasury_general_account": "WDTGAL",
    "dxy_proxy": "TWEXBMTH",
    "gold": "GOLDPMGBD228NLBM",
    "oil": "DCOILWTICO",
    "copper": "PCOPPUSDM",
    "gdp": "GDP",
    "wilshire_5000": "WILL5000INDFC",
    "net_equity_issuance_proxy": "NCBEILQ027S",
    "market_value_equity_proxy": "NCBCEBQ027S",
}

YAHOO_FEATURE_SYMBOLS = {
    "vix": "^VIX",
    "vix3m": "^VIX3M",
    "russell_2000": "^RUT",
}

SHILLER_STOCK_MARKET_CSV_URL = (
    "https://posix4e.github.io/shiller_wrapper_data/data/stock_market_data.csv"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download public long-history daily data for annual SP500 search.")
    parser.add_argument("--output", default="data/public/sp500_annual_daily.csv")
    args = parser.parse_args()
    path = build_annual_public_data(args.output)
    print(path)
    return 0


def build_annual_public_data(output: str | Path) -> Path:
    base = download_yahoo_chart("^GSPC")
    base = base.rename(columns={"close": "sp500_close"})
    base["timestamp_dt"] = pd.to_datetime(base["timestamp"])
    base = base.set_index("timestamp_dt")
    close = base["sp500_close"].astype(float)
    panel = pd.DataFrame(
        {
            "timestamp": base["timestamp"],
            "open": base["open"],
            "high": base["high"],
            "low": base["low"],
            "close": close,
            "volume": base["volume"],
        },
        index=base.index,
    )
    for name, symbol in YAHOO_FEATURE_SYMBOLS.items():
        try:
            asset = download_yahoo_chart(symbol)
        except Exception:
            continue
        asset["timestamp_dt"] = pd.to_datetime(asset["timestamp"])
        aligned = asset.set_index("timestamp_dt").reindex(panel.index).ffill()
        series = aligned["close"].astype(float)
        panel[f"{name}_level"] = series
        panel[f"{name}_ret_12m"] = series.pct_change(252)
        panel[f"sp500_vs_{name}_ret_12m"] = close.pct_change(252) - series.pct_change(252)
    for name, series_id in FRED_ANNUAL_SERIES.items():
        try:
            series = _download_fred_series(series_id).shift(5, freq="B")
        except Exception:
            continue
        panel[name] = series.reindex(panel.index).ffill()
    try:
        valuation = _download_shiller_valuation_data().shift(5, freq="B")
        for column in ("cape", "earnings_yield", "dividend_yield"):
            if column in valuation:
                panel[column] = valuation[column].reindex(panel.index).ffill()
    except Exception:
        pass
    if {"hy_oas", "ig_oas"}.issubset(panel.columns):
        panel["hy_minus_ig_oas"] = panel["hy_oas"] - panel["ig_oas"]
    if {"baa", "aaa"}.issubset(panel.columns):
        panel["baa_aaa_spread"] = panel["baa"] - panel["aaa"]
    if {"yield_10y", "yield_2y"}.issubset(panel.columns):
        panel["yield_10y_minus_2y"] = panel["yield_10y"] - panel["yield_2y"]
    if {"wilshire_5000", "gdp"}.issubset(panel.columns):
        panel["market_cap_to_gdp"] = panel["wilshire_5000"] / panel["gdp"]
    if {"net_equity_issuance_proxy", "market_value_equity_proxy"}.issubset(panel.columns):
        panel["buyback_yield"] = -panel["net_equity_issuance_proxy"] / panel["market_value_equity_proxy"]
    panel = panel.reset_index(drop=True)
    return write_public_data(panel, output)


def _download_shiller_valuation_data() -> pd.DataFrame:
    request = Request(SHILLER_STOCK_MARKET_CSV_URL, headers={"User-Agent": "trading-lab-annual-valuation"})
    with urlopen(request, timeout=30) as response:
        raw = pd.read_csv(response)
    date_column = "date_string" if "date_string" in raw.columns else "date"
    if date_column not in raw.columns:
        raise ValueError("Shiller valuation data missing date column")
    index = pd.to_datetime(raw[date_column], errors="coerce")
    frame = pd.DataFrame(index=index)
    if "cape" in raw:
        frame["cape"] = pd.to_numeric(raw["cape"], errors="coerce")
    if {"earnings", "sp500"}.issubset(raw.columns):
        earnings = pd.to_numeric(raw["earnings"], errors="coerce")
        price = pd.to_numeric(raw["sp500"], errors="coerce")
        frame["earnings_yield"] = earnings / price
        frame["pe_ttm"] = price / earnings.replace(0, pd.NA)
        frame["eps_ttm"] = earnings
    if {"dividend", "sp500"}.issubset(raw.columns):
        dividend = pd.to_numeric(raw["dividend"], errors="coerce")
        price = pd.to_numeric(raw["sp500"], errors="coerce")
        frame["dividend_yield"] = dividend / price
    return frame.replace([float("inf"), float("-inf")], pd.NA).dropna(how="all").sort_index()


if __name__ == "__main__":
    raise SystemExit(main())
