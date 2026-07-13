"""QLD/TQQQ 투자 전략 정의.

모든 전략은 일별 목표 비중 DataFrame(컬럼: QLD, TQQQ, cash)을 반환한다.
신호는 전일 종가 기준으로 계산해 다음 날 반영(선행편향 방지)한다.
"""
import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _shift(w: pd.DataFrame) -> pd.DataFrame:
    """신호 익일 반영."""
    return w.shift(1).fillna(0.0)


def buy_and_hold(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=df.index, columns=["QLD", "TQQQ", "cash"])
    w[asset] = 1.0
    return _shift(w)


def sma_timing(df: pd.DataFrame, asset: str = "TQQQ", window: int = 200) -> pd.DataFrame:
    """QQQ가 이동평균 위면 레버리지 보유, 아래면 현금."""
    sig = df["QQQ"] > df["QQQ"].rolling(window).mean()
    w = pd.DataFrame(0.0, index=df.index, columns=["QLD", "TQQQ", "cash"])
    w.loc[sig, asset] = 1.0
    w.loc[~sig, "cash"] = 1.0
    return _shift(w)


def vol_target(df: pd.DataFrame, asset: str = "TQQQ", target: float = 0.35,
               window: int = 20) -> pd.DataFrame:
    """레버리지 자산의 실현변동성이 목표를 넘으면 비중 축소."""
    ret = df[f"{asset}_syn"].pct_change()
    rv = ret.rolling(window).std() * np.sqrt(TRADING_DAYS)
    frac = (target / rv).clip(upper=1.0).fillna(0.0)
    w = pd.DataFrame(0.0, index=df.index, columns=["QLD", "TQQQ", "cash"])
    w[asset] = frac
    w["cash"] = 1.0 - frac
    return _shift(w)


def regime_rotation(df: pd.DataFrame, window: int = 200,
                    vol_window: int = 20, vol_split: float = 0.25) -> pd.DataFrame:
    """추세 + 변동성 레짐 로테이션.

    - QQQ > 200일선 & 저변동성 → TQQQ (공격)
    - QQQ > 200일선 & 고변동성 → QLD (완충)
    - QQQ < 200일선            → 현금 (방어)
    """
    trend = df["QQQ"] > df["QQQ"].rolling(window).mean()
    rv = df["QQQ"].pct_change().rolling(vol_window).std() * np.sqrt(TRADING_DAYS)
    calm = rv < vol_split
    w = pd.DataFrame(0.0, index=df.index, columns=["QLD", "TQQQ", "cash"])
    w.loc[trend & calm, "TQQQ"] = 1.0
    w.loc[trend & ~calm, "QLD"] = 1.0
    w.loc[~trend, "cash"] = 1.0
    return _shift(w)


def fixed_mix(df: pd.DataFrame, qld: float = 0.5, tqqq: float = 0.5) -> pd.DataFrame:
    """QLD/TQQQ 고정 배분(매일 리밸런싱 근사)."""
    w = pd.DataFrame(0.0, index=df.index, columns=["QLD", "TQQQ", "cash"])
    w["QLD"], w["TQQQ"] = qld, tqqq
    return _shift(w)
