"""TQQQ 60 / SGOV 40 디시전 트리 vs 현행 모델 — 27년 홀딩스 기반 비교.

제안 전략 규칙 (원문 그대로):
  목표배분  TQQQ 60 / SGOV 40
  적립규칙  신규 자금은 ① SGOV가 40% 될 때까지 SGOV → ② TQQQ가 목표될 때까지 TQQQ
            → ③ 둘 다 목표면 60/40 분할
  익절규칙  TQQQ가 포트폴리오의 65%를 넘으면 초과분을 SGOV로 트림
  낙폭규칙  QQQ가 52주 고점 대비 -10 / -15 / -20 / -25%일 때
            각 구간마다 SGOV의 12.5%씩 TQQQ로 투입 (사이클당 1회)
  리셋규칙  TQQQ가 20주 고점 밴드 위에서 마감하거나 목표비중을 회복하면 낙폭매수 중단,
            신규 자금·트림으로 SGOV를 먼저 채운 뒤 TQQQ 추가

원문이 애매해 해석이 필요했던 부분(파라미터로 노출):
  * "trim the excess" → 65% 초과분을 목표 60%까지 되돌리는 것으로 구현(trim_to=0.60).
  * "12.5% of SGOV" → 문자 그대로면 트리거 시점 SGOV 잔고의 12.5%(ladder="balance").
    4번 다 발동해도 SGOV의 41%만 소진돼 "final 12.5%"라는 표현과 어긋나므로,
    사이클 시작 SGOV를 4등분해 전액 소진하는 해석(ladder="quarter")도 함께 계산한다.
  * 리셋은 "recovers"를 반영해 낙폭이 -10%보다 얕은 상태에서
    (TQQQ 20주 신고가 or TQQQ 비중 ≥ 60%)일 때만 발동. 이 조건이 없으면
    낙폭매수 직후 비중이 60%를 넘어 같은 구간을 매일 재발동하는 루프가 생긴다.

프록시: SGOV → 13주 T-bill(cash_ret), QQQM → QQQ, QLD/TQQQ → 합성 시리즈.
비교 대상 현행 모델은 compare_vercel.py의 sim_ours와 동일한 상태머신이다.

사용법:  python3 compare_sgov.py             # 일시금 + 적립식 비교표
        python3 compare_sgov.py --selftest  # 합성 랜덤 시세로 로직 점검(데이터 불필요)
"""
import sys

import numpy as np
import pandas as pd

import backtest as bt

COST = 5 / 10000     # 편도 거래비용 5bp
TD = 252
MONTHLY = 0.005      # 적립식 시나리오: 매월 초기자본의 0.5% 납입(연 6%)

PERIODS = {"전기간": (None, None), "폭락기 99-09": ("1999", "2009"),
           "2010s": ("2010", "2019"), "2020s": ("2020", None)}


# --------------------------------------------------------------------------
# 데이터
# --------------------------------------------------------------------------
def prepare(df: pd.DataFrame):
    rets = pd.DataFrame({
        "QLD": df["QLD_syn"].pct_change().fillna(0),
        "QQQ": df["QQQ"].pct_change().fillna(0),
        "TQQQ": df["TQQQ_syn"].pct_change().fillna(0),
        "cash": df["cash_ret"],
    })
    ind = pd.DataFrame({
        "qqq": df["QQQ"],
        "sma200": df["QQQ"].rolling(200).mean(),
        "high52": df["QQQ"].rolling(252, min_periods=1).max(),
        "tqqq": df["TQQQ_syn"],
        "tqqq_high20w": df["TQQQ_syn"].rolling(100, min_periods=1).max(),
    })
    idx = df.index
    n = len(idx)
    month_end = np.array([i + 1 == n or idx[i + 1].month != idx[i].month
                          for i in range(n)])
    year_end = np.array([i + 1 == n or idx[i + 1].year != idx[i].year
                         for i in range(n)])
    return idx, rets, ind, month_end, year_end


def load():
    import data
    df = data.build_dataset()
    # data.py는 ^IRX 다운로드가 실패해도 조용히 연 4% 고정으로 대체한다.
    # 그 경우 TQQQ 합성 차입비용이 4%p 가까이 틀어져 표 전체가 바뀌므로 여기서 막는다.
    cash = df["cash_ret"]
    if cash.sum() <= 0 or (cash == 0).mean() > 0.05:
        raise RuntimeError(
            "^IRX(13주 국채금리) 데이터가 비어 있다. 합성 레버리지 비용이 "
            "연 4% 고정값으로 대체돼 결과가 왜곡되므로 중단한다. 다시 실행할 것.")
    print(f"현금금리(^IRX) 평균 {cash.mean() * TD:.2%}, 결측 {(cash == 0).mean():.2%}")
    return prepare(df)


def synthetic(n_years: int = 27, seed: int = 0):
    """--selftest용 합성 시세(기하 브라운 운동 + 급락 구간)."""
    rng = np.random.default_rng(seed)
    n = n_years * TD
    idx = pd.bdate_range("1999-03-10", periods=n)
    r = rng.normal(0.10 / TD, 0.28 / np.sqrt(TD), n)
    for start in (300, 2200, 5300):                     # 인위적 폭락 구간
        r[start:start + 250] -= 0.004
    qqq = pd.Series(100 * np.exp(np.cumsum(r)), index=idx)
    irx = pd.Series(3.0, index=idx)
    import data as d
    df = pd.DataFrame({"QQQ": qqq})
    df["QLD_syn"] = d.synth_leverage(qqq, irx, 2)
    df["TQQQ_syn"] = d.synth_leverage(qqq, irx, 3)
    df["cash_ret"] = 0.03 / TD
    return prepare(df)


# --------------------------------------------------------------------------
# 좌수(unit) 장부 — 적립금이 있어도 시간가중 수익률을 볼 수 있게 한다
# --------------------------------------------------------------------------
class Book:
    def __init__(self, value: float, n: int):
        self.units = 1.0
        self.unit_val = value
        self.contributed = value
        self.flows = [(0, -value)]      # IRR용 현금흐름(투자=음수)
        self.units_hist = np.ones(n)

    def mark(self, value: float):
        """당일 수익 반영 후, 납입 전 좌수가격 갱신."""
        self.unit_val = value / self.units

    def add(self, amount: float, i: int):
        self.units += amount / self.unit_val
        self.contributed += amount
        self.flows.append((i, -amount))

    def stamp(self, i: int):
        self.units_hist[i] = self.units


def trade_to(h, target_w, tot):
    """목표 비중으로 매매, 비용 차감한 새 보유액 반환."""
    target = target_w * tot
    tot -= np.abs(target - h).sum() * COST
    return target_w * tot, tot


def irr(flows, n_days, final_value):
    """일간 현금흐름 → 연율 IRR (이분법)."""
    cf = list(flows) + [(n_days - 1, final_value)]

    def npv(rate):
        d = (1 + rate) ** (1 / TD)
        return sum(c / d ** i for i, c in cf)

    lo, hi = -0.99, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if npv(mid) > 0 else (lo, mid)
    return (lo + hi) / 2


def summarize(label, curve, book, n):
    """시간가중(좌수가격) 지표 + 금액가중(IRR) 결과."""
    unit = curve / book.units_hist
    out = {"모델": label}
    for name, (a, b) in PERIODS.items():
        c = unit.loc[a:b] if (a or b) else unit
        c = c / c.iloc[0]
        m = bt.metrics(c)
        out[f"CAGR {name}"] = m["CAGR"]
        if name == "전기간":
            out["MDD"] = m["MDD"]
            out["샤프"] = m["샤프"]
            out["Calmar"] = m["CAGR"] / abs(m["MDD"]) if m["MDD"] < 0 else np.nan
    out["납입원금"] = book.contributed
    out["최종자산"] = float(curve.iloc[-1])
    out["원금배수"] = out["최종자산"] / book.contributed
    out["IRR"] = irr(book.flows, n, out["최종자산"])
    return out


# --------------------------------------------------------------------------
# 제안 전략: TQQQ 60 / SGOV 40 디시전 트리
# --------------------------------------------------------------------------
def sim_sgov(idx, rets, ind, month_end, contrib=0.0, trim_band=0.65,
             trim_to=0.60, ladder="balance", tranche=0.125,
             levels=(-0.10, -0.15, -0.20, -0.25), sma_gate=False):
    r = rets[["TQQQ", "cash"]].to_numpy()
    qqq, s200 = ind["qqq"].to_numpy(), ind["sma200"].to_numpy()
    high52 = ind["high52"].to_numpy()
    tqqq, tq_hi = ind["tqqq"].to_numpy(), ind["tqqq_high20w"].to_numpy()
    n = len(idx)

    h = np.array([0.60, 0.40])          # TQQQ, SGOV
    book = Book(h.sum(), n)
    fired, cycle_sgov = set(), h[1]
    val = np.zeros(n)
    n_ladder = n_trim = 0

    for i in range(n):
        h = h * (1 + r[i])
        tot = h.sum()
        book.mark(tot)

        # ① 적립: SGOV 40% 회복 우선 → TQQQ 목표 → 둘 다 차면 60/40
        if contrib and month_end[i]:
            book.add(contrib, i)
            after = tot + contrib
            to_sgov = min(contrib, max(0.0, 0.40 * after - h[1]))
            rest = contrib - to_sgov
            to_tqqq = min(rest, max(0.0, 0.60 * after - h[0]))
            rest -= to_tqqq
            to_tqqq += 0.60 * rest
            to_sgov += 0.40 * rest
            h[0] += to_tqqq * (1 - COST)
            h[1] += to_sgov
            tot = h.sum()
        book.stamp(i)

        if np.isnan(s200[i]):
            val[i] = tot
            continue

        # ② 익절 트림: TQQQ > 65% → 목표 60%까지 SGOV로
        if h[0] > trim_band * tot:
            amt = h[0] - trim_to * tot
            h[0] -= amt
            h[1] += amt * (1 - COST)
            tot = h.sum()
            n_trim += 1

        # ③ 낙폭 사다리: 구간별 SGOV 12.5%씩 TQQQ 투입(사이클당 1회)
        # sma_gate=True면 200일선 아래에서는 발동을 보류하고, 회복 후 밀린 구간을 한 번에 집행
        dd = qqq[i] / high52[i] - 1
        gate_open = (not sma_gate) or qqq[i] >= s200[i]
        for lv in levels:
            if dd <= lv and lv not in fired and gate_open:
                fired.add(lv)
                amt = tranche * h[1] if ladder == "balance" else min(0.25 * cycle_sgov, h[1])
                if amt > 0:
                    h[1] -= amt
                    h[0] += amt * (1 - COST)
                    n_ladder += 1
        tot = h.sum()

        # ④ 리셋: 낙폭에서 벗어난 뒤 TQQQ 20주 신고가 or 목표비중 회복
        if dd > levels[0] and (tqqq[i] >= tq_hi[i] or h[0] >= 0.60 * tot):
            fired.clear()
            cycle_sgov = h[1]

        val[i] = tot

    return pd.Series(val, index=idx), book, {"사다리매수": n_ladder, "트림": n_trim}


# --------------------------------------------------------------------------
# 현행 모델: QQQM 50 / QLD 50 + 낙폭 -10%에서 TQQQ 30%p 위성
# --------------------------------------------------------------------------
def sim_ours(idx, rets, ind, month_end, year_end, contrib=0.0):
    BASE = np.array([0.5, 0.5, 0.0, 0.0])    # QLD, QQQM, TQQQ, cash
    ENTRY = np.array([0.5, 0.2, 0.3, 0.0])
    STOP = np.array([0.0, 0.3, 0.0, 0.7])
    TARGET = {0: BASE, 1: ENTRY, 2: STOP}
    r = rets[["QLD", "QQQ", "TQQQ", "cash"]].to_numpy()
    qqq, s200 = ind["qqq"].to_numpy(), ind["sma200"].to_numpy()
    high52, tqqq = ind["high52"].to_numpy(), ind["tqqq"].to_numpy()
    n = len(idx)

    h = BASE.copy()
    book = Book(h.sum(), n)
    state, th, pk = 0, np.nan, np.nan
    val = np.zeros(n)
    n_entry = n_stop = 0

    for i in range(n):
        h = h * (1 + r[i])
        tot = h.sum()
        book.mark(tot)

        # 적립: 현행 모델엔 적립규칙이 없으므로 현재 상태의 목표배분대로 납입
        if contrib and month_end[i]:
            book.add(contrib, i)
            h += contrib * TARGET[state] * (1 - COST)
            tot = h.sum()
        book.stamp(i)

        if not np.isnan(s200[i]):
            dd = qqq[i] / high52[i] - 1
            prev = state
            if state == 0:
                if dd <= -0.10 and qqq[i] >= s200[i]:
                    state, th, pk = 1, high52[i], tqqq[i]
                    n_entry += 1
            elif state == 1:
                pk = max(pk, tqqq[i])
                if qqq[i] >= th:
                    state = 0
                elif tqqq[i] / pk - 1 <= -0.30:
                    state = 2
                    n_stop += 1
            elif state == 2:
                if qqq[i] >= th:
                    state = 0
            if state != prev:
                h, tot = trade_to(h, TARGET[state], tot)
            elif state == 0 and year_end[i]:
                h, tot = trade_to(h, BASE, tot)      # 연 1회 리밸런싱
        val[i] = tot

    return pd.Series(val, index=idx), book, {"TQQQ진입": n_entry, "손절": n_stop}


# --------------------------------------------------------------------------
# 참고선: 고정비중
# --------------------------------------------------------------------------
def sim_static(idx, rets, ind, month_end, contrib=0.0, w_tqqq=0.6, rebal=True):
    r = rets[["TQQQ", "cash"]].to_numpy()
    W = np.array([w_tqqq, 1 - w_tqqq])
    h = W.copy()
    n = len(idx)
    book = Book(h.sum(), n)
    val = np.zeros(n)
    for i in range(n):
        h = h * (1 + r[i])
        tot = h.sum()
        book.mark(tot)
        if contrib and month_end[i]:
            book.add(contrib, i)
            h += contrib * W * (1 - COST)
            tot = h.sum()
        book.stamp(i)
        if rebal and month_end[i] and 0 < w_tqqq < 1:
            h, tot = trade_to(h, W, tot)
        val[i] = tot
    return pd.Series(val, index=idx), book, {}


def run_all(idx, rets, ind, month_end, year_end, contrib):
    c1, b1, s1 = sim_sgov(idx, rets, ind, month_end, contrib)
    c2, b2, s2 = sim_sgov(idx, rets, ind, month_end, contrib, ladder="quarter")
    c3, b3, s3 = sim_sgov(idx, rets, ind, month_end, contrib, sma_gate=True)
    c4, b4, s4 = sim_ours(idx, rets, ind, month_end, year_end, contrib)
    c5, b5, _ = sim_static(idx, rets, ind, month_end, contrib, 0.6)
    c6, b6, _ = sim_static(idx, rets, ind, month_end, contrib, 1.0, rebal=False)
    return [
        (f"제안 60/40 (12.5%×SGOV잔고, 매수 {s1['사다리매수']}회/트림 {s1['트림']}회)", c1, b1),
        (f"제안 60/40 (SGOV 4등분 전액소진, 매수 {s2['사다리매수']}회)", c2, b2),
        (f"제안 60/40 + 200일선 게이트 (매수 {s3['사다리매수']}회)", c3, b3),
        (f"현행 QQQM50/QLD50+TQQQ위성 (진입 {s4['TQQQ진입']}회/손절 {s4['손절']}회)", c4, b4),
        ("[참고] TQQQ60/SGOV40 월말 고정 리밸런싱", c5, b5),
        ("[참고] TQQQ 100% 보유", c6, b6),
    ]


def report(idx, rets, ind, month_end, year_end):
    pd.options.display.float_format = "{:.3f}".format
    pd.options.display.width = 250
    print(f"데이터 구간: {idx[0].date()} ~ {idx[-1].date()} ({len(idx)}일)")
    print("SGOV=13주 T-bill, QQQM=QQQ, QLD/TQQQ=합성, 거래비용 5bp")
    print("CAGR/MDD/샤프는 좌수가격(시간가중), IRR은 납입 반영(금액가중)\n")

    for title, contrib in [("A. 일시금 (적립 없음, 순수 규칙 비교)", 0.0),
                           (f"B. 적립식 (매월 초기자본의 {MONTHLY:.1%} 납입)", MONTHLY)]:
        print("=" * 130)
        print(title)
        print("=" * 130)
        rows = [summarize(label, curve, book, len(idx))
                for label, curve, book in
                run_all(idx, rets, ind, month_end, year_end, contrib)]
        t = pd.DataFrame(rows).set_index("모델")
        cols = ["CAGR 전기간", "MDD", "샤프", "Calmar",
                "CAGR 폭락기 99-09", "CAGR 2010s", "CAGR 2020s"]
        if contrib:
            cols += ["납입원금", "최종자산", "원금배수", "IRR"]
        print(t[cols].to_string(), "\n")

    # 위기 구간별 낙폭 비교
    print("=" * 130)
    print("C. 주요 위기 구간 손실 (일시금 기준, 구간 내 저점까지 누적수익률)")
    print("=" * 130)
    crises = {"닷컴 2000-03~2002-10": ("2000-03-01", "2002-10-31"),
              "금융위기 2007-10~2009-03": ("2007-10-01", "2009-03-31"),
              "코로나 2020-02~2020-03": ("2020-02-01", "2020-03-31"),
              "2022 긴축 2022-01~2022-12": ("2022-01-01", "2022-12-31")}
    rows = {}
    for label, curve, _ in run_all(idx, rets, ind, month_end, year_end, 0.0):
        row = {}
        for name, (a, b) in crises.items():
            c = curve.loc[a:b]
            row[name] = (c.min() / c.iloc[0] - 1) if len(c) else np.nan
        rows[label] = row
    print(pd.DataFrame(rows).T.to_string())


def main():
    if "--selftest" in sys.argv:
        print("[selftest] 합성 시세로 로직 점검 (수치는 의미 없음)\n")
        report(*synthetic())
        return
    report(*load())


if __name__ == "__main__":
    main()
