"""QQQ 기반 레버리지 시리즈 합성 및 데이터 로딩."""
import numpy as np
import pandas as pd
import yfinance as yf

CACHE = "prices.parquet"

# 합성 파라미터: 일일 리밸런싱 레버리지 ETF 근사
# 총보수 + 스왑/차입 비용을 연율로 차감
EXPENSE = {2: 0.0095, 3: 0.0086}          # QLD 0.95%, TQQQ 0.86%
BORROW_SPREAD = 0.005                      # 차입 스프레드 가정

TRADING_DAYS = 252


def download(force: bool = False) -> pd.DataFrame:
    """QQQ, QLD, TQQQ 수정종가와 13주 국채금리(^IRX)를 받아온다."""
    import os
    if not force and os.path.exists(CACHE):
        return pd.read_parquet(CACHE)
    px = yf.download(["QQQ", "QQQM", "QLD", "TQQQ", "^IRX"], start="1999-03-10",
                     auto_adjust=True, progress=False)["Close"]
    px = px.rename(columns={"^IRX": "IRX"})
    px.to_parquet(CACHE)
    return px


def synth_leverage(qqq: pd.Series, irx: pd.Series, lev: int) -> pd.Series:
    """QQQ 일일수익률로 lev배 레버리지 ETF 순자산 시리즈를 합성한다."""
    r = qqq.pct_change()
    cash_rate = (irx.ffill() / 100).reindex(r.index).ffill().fillna(0.04)
    daily_cost = (EXPENSE[lev] + (lev - 1) * (cash_rate + BORROW_SPREAD)) / TRADING_DAYS
    lr = lev * r - daily_cost
    lr.iloc[0] = 0.0
    return (1 + lr).cumprod()


def build_dataset(force: bool = False) -> pd.DataFrame:
    """백테스트용 데이터셋: QQQ, 합성/실제 QLD·TQQQ, 현금수익률."""
    px = download(force)
    qqq = px["QQQ"].dropna()
    irx = px["IRX"]
    df = pd.DataFrame(index=qqq.index)
    df["QQQ"] = qqq
    df["QLD_syn"] = synth_leverage(qqq, irx, 2)
    df["TQQQ_syn"] = synth_leverage(qqq, irx, 3)
    df["QLD"] = px["QLD"]
    df["TQQQ"] = px["TQQQ"]
    df["QQQM"] = px["QQQM"]  # 2020-10 상장, 표시용
    df["cash_ret"] = (irx.ffill() / 100 / TRADING_DAYS).reindex(df.index).fillna(0)
    return df


def validate_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    """실제 ETF 존재 구간에서 합성 시리즈와의 연율 수익률 괴리를 점검."""
    out = {}
    for real, syn in [("QLD", "QLD_syn"), ("TQQQ", "TQQQ_syn")]:
        sub = df[[real, syn]].dropna()
        rr = sub[real].pct_change().dropna()
        rs = sub[syn].pct_change().dropna()
        corr = rr.corr(rs)
        cagr_diff = ((sub[real].iloc[-1] / sub[real].iloc[0]) ** (TRADING_DAYS / len(sub))
                     - (sub[syn].iloc[-1] / sub[syn].iloc[0]) ** (TRADING_DAYS / len(sub)))
        out[real] = {"일일수익률 상관": corr, "CAGR 괴리": cagr_diff, "표본일수": len(sub)}
    return pd.DataFrame(out).T
