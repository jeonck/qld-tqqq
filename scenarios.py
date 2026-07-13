"""QQQM+QLD 기본 보유 → TQQQ 진입/익절/방어 시나리오 그리드 서치.

전략 상태머신 (일별, 신호는 전일 종가 → 익일 반영):

  [기본]   QQQ(=QQQM 대용) base_qqq + QLD (1-base_qqq) 보유
  [진입]   QQQ가 52주 고점 대비 dip_entry 이상 하락하면
           포트폴리오의 tqqq_frac 만큼 TQQQ로 전환 (QLD부터 재원 사용)
           allow_below_sma=False면 QQQ<200일선일 때는 진입 대기
  [익절]   QQQ가 진입 시점의 52주 고점을 회복하면 TQQQ 전량 익절 → 기본 배분 복귀
  [손절]   trail_stop 설정 시 TQQQ 진입 후 TQQQ가 고점 대비 trail_stop 하락하면
           청산 → 현금 (QQQ가 52주 고점 회복해야 재진입 리셋)
  [방어]   defense=True면 QQQ<200일선 구간에서 QLD 몫을 현금으로 대피
           (TQQQ 딥매수 포지션은 별도 규칙으로만 관리)
"""
import itertools
import numpy as np
import pandas as pd

import backtest as bt

TRADING_DAYS = 252


def escalation_weights(df: pd.DataFrame, base_qqq: float, dip_entry: float,
                       tqqq_frac: float, allow_below_sma: bool,
                       trail_stop: float | None, defense: bool) -> pd.DataFrame:
    qqq = df["QQQ"].to_numpy()
    tqqq_px = df["TQQQ_syn"].to_numpy()
    sma = df["QQQ"].rolling(200).mean().to_numpy()
    high52 = df["QQQ"].rolling(252, min_periods=1).max().to_numpy()
    n = len(df)

    w = np.zeros((n, 4))  # QQQ, QLD, TQQQ, cash
    IN_TQQQ, STOPPED, BASE = 1, 2, 0
    state = BASE
    target_high = np.nan   # 익절 목표(진입 시점 52주 고점)
    tqqq_peak = np.nan     # 트레일링 스탑용 TQQQ 고점

    for i in range(n):
        if not np.isnan(sma[i]):
            dd = qqq[i] / high52[i] - 1
            above = qqq[i] >= sma[i]

            if state == BASE:
                if dd <= dip_entry and (above or allow_below_sma):
                    state = IN_TQQQ
                    target_high = high52[i]
                    tqqq_peak = tqqq_px[i]
            elif state == IN_TQQQ:
                tqqq_peak = max(tqqq_peak, tqqq_px[i])
                if qqq[i] >= target_high:                      # 익절
                    state = BASE
                elif trail_stop is not None and tqqq_px[i] / tqqq_peak - 1 <= trail_stop:
                    state = STOPPED                            # 손절
            elif state == STOPPED:
                if qqq[i] >= target_high:                      # 회복 후 리셋
                    state = BASE

        # 상태 → 비중
        qld_w = 1.0 - base_qqq
        if state == IN_TQQQ:
            take = min(tqqq_frac, 1.0)
            from_qld = min(take, qld_w)
            from_qqq = take - from_qld
            w[i] = [base_qqq - from_qqq, qld_w - from_qld, take, 0.0]
        elif state == STOPPED:
            w[i] = [base_qqq, 0.0, 0.0, qld_w]                 # 손절 재원은 현금 대기
        else:
            w[i] = [base_qqq, qld_w, 0.0, 0.0]

        # 방어: 200일선 아래면 QLD → 현금
        if defense and not np.isnan(sma[i]) and qqq[i] < sma[i] and w[i, 1] > 0:
            w[i, 3] += w[i, 1]
            w[i, 1] = 0.0

    out = pd.DataFrame(w, index=df.index, columns=["QQQ", "QLD", "TQQQ", "cash"])
    return out.shift(1).fillna(0.0)


PERIODS = {
    "전기간": (None, None),
    "1999-2009 (2번의 폭락)": ("1999", "2009"),
    "2010-2019 (강세장)": ("2010", "2019"),
    "2020-현재": ("2020", None),
}


def sub_metrics(curve: pd.Series) -> dict:
    row = {}
    for name, (a, b) in PERIODS.items():
        c = curve.loc[a:b] if (a or b) else curve
        c = c / c.iloc[0]
        m = bt.metrics(c)
        row[f"CAGR {name}"] = m["CAGR"]
        if name == "전기간":
            row["MDD"] = m["MDD"]
            row["샤프"] = m["샤프"]
            row["Calmar"] = m["CAGR"] / abs(m["MDD"]) if m["MDD"] < 0 else np.nan
    return row


def grid_search(df: pd.DataFrame) -> pd.DataFrame:
    grid = list(itertools.product(
        [0.3, 0.5, 0.7],                       # base_qqq (나머지 QLD)
        [-0.10, -0.15, -0.20, -0.25, -0.30],   # dip_entry
        [0.3, 0.5, 0.7, 1.0],                  # tqqq_frac
        [True, False],                         # allow_below_sma
        [None, -0.20, -0.30],                  # trail_stop
        [False, True],                         # defense
    ))
    rows = []
    for bq, de, tf, ab, ts, dfn in grid:
        wts = escalation_weights(df, bq, de, tf, ab, ts, dfn)
        curve = bt.run(df, wts)
        row = {"QQQM비중": bq, "진입낙폭": de, "TQQQ비중": tf,
               "선하진입": ab, "손절": ts if ts is not None else 0,
               "방어": dfn}
        row.update(sub_metrics(curve))
        rows.append(row)
    return pd.DataFrame(rows)
