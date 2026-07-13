"""백테스트 엔진과 성과지표."""
import numpy as np
import pandas as pd

TRADING_DAYS = 252
COST_BPS = 5  # 편도 거래비용 0.05% (수수료+슬리피지 가정)


def run(df: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    """비중 시리즈로 전략 누적자산 곡선을 계산한다. 합성 시리즈 사용(전 기간)."""
    rets = pd.DataFrame({
        "QQQ": df["QQQ"].pct_change().fillna(0),
        "QLD": df["QLD_syn"].pct_change().fillna(0),
        "TQQQ": df["TQQQ_syn"].pct_change().fillna(0),
        "cash": df["cash_ret"],
    })
    w = weights.reindex(df.index).fillna(0.0)
    gross = (w * rets).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(0)
    net = gross - turnover * COST_BPS / 10000
    return (1 + net).cumprod()


def metrics(curve: pd.Series, rf: float = 0.0) -> dict:
    curve = curve.dropna()
    ret = curve.pct_change().dropna()
    years = len(curve) / TRADING_DAYS
    cagr = curve.iloc[-1] ** (1 / years) - 1
    vol = ret.std() * np.sqrt(TRADING_DAYS)
    dd = curve / curve.cummax() - 1
    downside = ret[ret < 0].std() * np.sqrt(TRADING_DAYS)
    return {
        "CAGR": cagr,
        "연변동성": vol,
        "샤프": (cagr - rf) / vol if vol > 0 else np.nan,
        "소르티노": (cagr - rf) / downside if downside > 0 else np.nan,
        "MDD": dd.min(),
        "최종배수": curve.iloc[-1],
    }


def compare(df: pd.DataFrame, strategy_map: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """전략별 자산곡선과 지표 테이블을 반환."""
    curves, rows = {}, {}
    for name, weight_fn in strategy_map.items():
        curve = run(df, weight_fn(df))
        curves[name] = curve
        rows[name] = metrics(curve)
    return pd.DataFrame(curves), pd.DataFrame(rows).T
