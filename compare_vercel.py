"""qld-qqqm.vercel.app 전략 vs 현행 모델 — 27년 홀딩스 기반 비교.

vercel 전략 규칙 (사이트 명세 그대로):
  A 상승추세(QQQ가 50일선·200일선 위): QLD 47.5 / QQQM 20 / GPIQ 10 / SGOV 22.5, 월말 리밸런싱
  B 단기조정(50일선 아래): 동결 (매수·매도 금지, 드리프트 방치)
  C 추세훼손(월말 종가 < 200일선 -2%): QLD 15 / QQQM 19 / GPIQ 7.5 / SGOV 58.5로 축소
  D 회복(주간 종가 2주 연속 200일선 위): A 배분 복귀
  밴드 매수(사상최고가 대비, 사이클당 1회, 상시 발동):
    -8%: QQQM +3.75%p / -15%: QQQM +3.75%p + QLD +6.25%p
    -25%: QLD +12.5%p / -35%: 잔여 현금 절반 → QLD 6 : QQQM 4
프록시: QQQM→QQQ, SGOV→T-bill(cash_ret), GPIQ→0.7×QQQ - 연 3.5% 드래그(QYLD 근사)
"""
import numpy as np
import pandas as pd

import data
import backtest as bt

COST = 5 / 10000
TD = 252


def metrics(curve: pd.Series, label: str):
    out = {"모델": label}
    periods = {"전기간": (None, None), "폭락기 99-09": ("1999", "2009"),
               "2010s": ("2010", "2019"), "2020s": ("2020", None)}
    for name, (a, b) in periods.items():
        c = curve.loc[a:b] if (a or b) else curve
        c = c / c.iloc[0]
        m = bt.metrics(c)
        out[f"CAGR {name}"] = m["CAGR"]
        if name == "전기간":
            out["MDD"] = m["MDD"]
            out["샤프"] = m["샤프"]
    return out


def load():
    df = data.build_dataset()
    rets = pd.DataFrame({
        "QLD": df["QLD_syn"].pct_change().fillna(0),
        "QQQ": df["QQQ"].pct_change().fillna(0),
        "GPIQ": 0.7 * df["QQQ"].pct_change().fillna(0) - 0.035 / TD,
        "TQQQ": df["TQQQ_syn"].pct_change().fillna(0),
        "cash": df["cash_ret"],
    })
    ind = pd.DataFrame({
        "qqq": df["QQQ"],
        "sma50": df["QQQ"].rolling(50).mean(),
        "sma200": df["QQQ"].rolling(200).mean(),
        "ath": df["QQQ"].cummax(),
        "high52": df["QQQ"].rolling(252, min_periods=1).max(),
        "tqqq": df["TQQQ_syn"],
    })
    idx = df.index
    month_end = np.array([i + 1 == len(idx) or idx[i + 1].month != idx[i].month
                          for i in range(len(idx))])
    week_end = np.array([i + 1 == len(idx) or idx[i + 1].week != idx[i].week
                         for i in range(len(idx))])
    return idx, rets, ind, month_end, week_end


def trade_to(h, target_w, tot):
    """목표 비중으로 매매, 비용 차감한 새 보유액 반환."""
    target = target_w * tot
    cost = np.abs(target - h).sum() * COST
    tot -= cost
    return target_w * (tot), tot


def sim_vercel(idx, rets, ind, month_end, week_end):
    # 자산 순서: QLD, QQQ(QQQM), GPIQ, cash
    A = np.array([0.475, 0.20, 0.10, 0.225])
    C = np.array([0.15, 0.19, 0.075, 0.585])
    r = rets[["QLD", "QQQ", "GPIQ", "cash"]].to_numpy()
    qqq = ind["qqq"].to_numpy(); s50 = ind["sma50"].to_numpy()
    s200 = ind["sma200"].to_numpy(); ath = ind["ath"].to_numpy()
    n = len(idx)
    h = A.copy()  # 1.0으로 시작
    mode = "A"; weeks_above = 0
    fired = set(); prev_ath = ath[0]
    curve = np.zeros(n)
    for i in range(n):
        h = h * (1 + r[i]); tot = h.sum()
        if np.isnan(s200[i]):
            curve[i] = tot; continue
        # 사이클 리셋: 신고가 갱신
        if ath[i] > prev_ath:
            fired.clear(); prev_ath = ath[i]
        dd = qqq[i] / ath[i] - 1
        # 밴드 매수 (상시, 사이클당 1회, 현금 한도 내)
        for band, buys in [(-0.08, [("QQQ", 0.0375)]),
                           (-0.15, [("QQQ", 0.0375), ("QLD", 0.0625)]),
                           (-0.25, [("QLD", 0.125)])]:
            if dd <= band and band not in fired:
                fired.add(band)
                for asset, frac in buys:
                    amt = min(frac * tot, h[3])
                    j = 0 if asset == "QLD" else 1
                    h[j] += amt * (1 - COST); h[3] -= amt
        if dd <= -0.35 and -0.35 not in fired:
            fired.add(-0.35)
            amt = h[3] * 0.5
            h[0] += amt * 0.6 * (1 - COST); h[1] += amt * 0.4 * (1 - COST)
            h[3] -= amt
        tot = h.sum()
        # 국면 판정
        if mode != "C":
            mode = "A" if (qqq[i] >= s50[i] and qqq[i] >= s200[i]) else "B"
        if month_end[i]:
            if qqq[i] < s200[i] * 0.98:
                if mode != "C":
                    mode = "C"; weeks_above = 0
                    h, tot = trade_to(h, C, tot)
            elif mode == "A":
                h, tot = trade_to(h, A, tot)  # 월말 리밸런싱
        if mode == "C" and week_end[i]:
            weeks_above = weeks_above + 1 if qqq[i] >= s200[i] else 0
            if weeks_above >= 2:
                mode = "A"
                h, tot = trade_to(h, A, tot)
        curve[i] = h.sum()
    return pd.Series(curve, index=idx)


def sim_ours(idx, rets, ind, month_end, week_end):
    # 자산 순서: QLD, QQQ(QQQM), TQQQ, cash
    BASE = np.array([0.5, 0.5, 0.0, 0.0])   # QQQM 50 / QLD 50
    ENTRY = np.array([0.5, 0.2, 0.3, 0.0])  # QQQM 20 / QLD 50 / TQQQ 30
    STOP = np.array([0.0, 0.3, 0.0, 0.7])   # QQQM 30 / 현금 70
    r = rets[["QLD", "QQQ", "TQQQ", "cash"]].to_numpy()
    qqq = ind["qqq"].to_numpy(); s200 = ind["sma200"].to_numpy()
    high52 = ind["high52"].to_numpy(); tqqq_px = ind["tqqq"].to_numpy()
    n = len(idx)
    h = BASE.copy(); state = 0; th = np.nan; pk = np.nan
    year_end = np.array([i + 1 == n or idx[i + 1].year != idx[i].year for i in range(n)])
    curve = np.zeros(n)
    for i in range(n):
        h = h * (1 + r[i]); tot = h.sum()
        if not np.isnan(s200[i]):
            dd = qqq[i] / high52[i] - 1; above = qqq[i] >= s200[i]
            prev = state
            if state == 0:
                if dd <= -0.10 and above:
                    state = 1; th = high52[i]; pk = tqqq_px[i]
            elif state == 1:
                pk = max(pk, tqqq_px[i])
                if qqq[i] >= th: state = 0
                elif tqqq_px[i] / pk - 1 <= -0.30: state = 2
            elif state == 2:
                if qqq[i] >= th: state = 0
            target = {0: BASE, 1: ENTRY, 2: STOP}[state]
            if state != prev:
                h, tot = trade_to(h, target, tot)
            elif state == 0 and year_end[i]:
                h, tot = trade_to(h, BASE, tot)  # 연 1회 리밸런싱
        curve[i] = h.sum()
    return pd.Series(curve, index=idx)


def main():
    idx, rets, ind, me, we = load()
    rows = [
        metrics(sim_ours(idx, rets, ind, me, we), "현행 모델 (50/50 + TQQQ 위성)"),
        metrics(sim_vercel(idx, rets, ind, me, we), "vercel 전략 (국면+밴드)"),
    ]
    out = pd.DataFrame(rows).set_index("모델")
    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 200
    print(out.to_string())
    print()
    # 2016~ (그쪽 백테스트 구간 재현 검증)
    for f, name in [(sim_ours, "현행"), (sim_vercel, "vercel")]:
        c = f(idx, rets, ind, me, we).loc["2016":"2025"]
        c = c / c.iloc[0]
        m = bt.metrics(c)
        print(f"2016-2025 {name}: {m['최종배수']:.2f}배, CAGR {m['CAGR']:.1%}, MDD {m['MDD']:.1%}")


if __name__ == "__main__":
    main()
