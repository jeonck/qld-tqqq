"""현재 시점 신호 상태 출력.

운용 조합(QQQM 30 / QLD 70 기본, 진입낙폭 -10% & 200일선 위에서
QQQM 전량 → TQQQ 전환(QLD 70/TQQQ 30), 익절 = 전고점 회복 시 QQQM 복귀,
손절 = TQQQ -30% 트레일링)의 상태머신을 오늘까지 실행해
현재 상태·목표 배분·트리거까지의 거리를 보여준다.

사용법:  python3 signal_now.py            # 캐시 데이터 사용
        python3 signal_now.py --refresh   # 최신 시세 다시 다운로드
"""
import sys
import numpy as np
import pandas as pd

import data
import scenarios as sc

# 운용 파라미터: 진입 시 QQQM 전량을 TQQQ로 전환 (QLD 70/TQQQ 30)
# 백테스트 1999~2026: CAGR 17.2%, MDD -57%, 샤프 0.60
PARAMS = dict(base_qqq=0.3, dip_entry=-0.10, tqqq_frac=0.3,
              allow_below_sma=False, trail_stop=-0.30, defense=False)

STATE_NAMES = {0: "기본 보유", 1: "TQQQ 진입 중", 2: "손절 후 현금 대기"}


def current_state(df: pd.DataFrame) -> dict:
    """escalation_weights와 동일한 상태머신을 돌려 마지막 날 상태를 반환."""
    qqq = df["QQQ"].to_numpy()
    tqqq_px = df["TQQQ_syn"].to_numpy()
    sma = df["QQQ"].rolling(200).mean().to_numpy()
    high52 = df["QQQ"].rolling(252, min_periods=1).max().to_numpy()

    state, target_high, tqqq_peak, entry_date = 0, np.nan, np.nan, None
    p = PARAMS
    for i in range(len(df)):
        if np.isnan(sma[i]):
            continue
        dd = qqq[i] / high52[i] - 1
        above = qqq[i] >= sma[i]
        if state == 0:
            if dd <= p["dip_entry"] and (above or p["allow_below_sma"]):
                state, target_high, tqqq_peak = 1, high52[i], tqqq_px[i]
                entry_date = df.index[i]
        elif state == 1:
            tqqq_peak = max(tqqq_peak, tqqq_px[i])
            if qqq[i] >= target_high:
                state, entry_date = 0, None
            elif tqqq_px[i] / tqqq_peak - 1 <= p["trail_stop"]:
                state = 2
        elif state == 2:
            if qqq[i] >= target_high:
                state, entry_date = 0, None

    i = len(df) - 1
    # QQQM 환산 비율: 신호는 QQQ(장기 데이터)로 계산하고 표시는 QQQM 가격으로
    qqqm = df["QQQM"].dropna()
    ratio = float(qqqm.iloc[-1] / df["QQQ"].loc[qqqm.index[-1]]) if len(qqqm) else np.nan
    return {
        "state": state, "date": df.index[i], "qqq": qqq[i], "sma": sma[i],
        "high52": high52[i], "dd": qqq[i] / high52[i] - 1,
        "above_sma": qqq[i] >= sma[i], "target_high": target_high,
        "tqqq_px": tqqq_px[i], "tqqq_peak": tqqq_peak, "entry_date": entry_date,
        "ratio": ratio,
    }


def main():
    force = "--refresh" in sys.argv
    df = data.build_dataset(force=force)
    s = current_state(df)
    p = PARAMS

    r = s["ratio"]  # QQQ → QQQM 가격 환산 (신호 계산은 QQQ 장기 데이터 기준)
    print(f"기준일: {s['date'].date()}  (데이터 갱신: python3 signal_now.py --refresh)")
    print("=" * 56)
    print(f"QQQM 종가       : {s['qqq'] * r:,.2f}")
    print(f"200일 이동평균   : {s['sma'] * r:,.2f}  "
          f"({'위 ✅' if s['above_sma'] else '아래 ❌'}, "
          f"이격 {s['qqq'] / s['sma'] - 1:+.1%})")
    print(f"52주 고점       : {s['high52'] * r:,.2f}")
    print(f"고점 대비 낙폭   : {s['dd']:+.1%}  (진입 기준 {p['dip_entry']:.0%})")
    print("=" * 56)

    state = s["state"]
    print(f"\n현재 상태: [{STATE_NAMES[state]}]\n")

    if state == 0:
        print("목표 배분: QQQM 30% / QLD 70%")
        gap = s["dd"] - p["dip_entry"]
        trigger_px = s["high52"] * (1 + p["dip_entry"]) * r
        if s["above_sma"]:
            print(f"→ TQQQ 진입 조건까지: QQQM이 {trigger_px:,.2f} 이하로 "
                  f"하락 시 (추가 {gap:+.1%})")
        else:
            print("→ 200일선 아래라 낙폭 조건을 채워도 진입 대기")
            print(f"  (200일선 {s['sma'] * r:,.2f} 회복 + 낙폭 {p['dip_entry']:.0%} 필요)")
    elif state == 1:
        dd_tq = s["tqqq_px"] / s["tqqq_peak"] - 1
        print("목표 배분: QLD 70% / TQQQ 30%  (QQQM 전량 → TQQQ)")
        print(f"진입일          : {s['entry_date'].date()}")
        print(f"익절 목표       : QQQM {s['target_high'] * r:,.2f} 회복 시 전량 익절 "
              f"(현재 {s['qqq'] / s['target_high'] - 1:+.1%})")
        print(f"손절 라인       : TQQQ 고점 대비 {p['trail_stop']:.0%} "
              f"(현재 {dd_tq:+.1%}, 여유 {dd_tq - p['trail_stop']:+.1%}p)")
    else:
        print("목표 배분: QQQM 30% / 현금 70%")
        print(f"→ QQQM이 {s['target_high'] * r:,.2f} (직전 전고점) 회복 시 "
              f"기본 배분으로 복귀 (현재 {s['qqq'] / s['target_high'] - 1:+.1%})")

    print("\n※ 백테스트 기반 리서치 도구이며 투자자문이 아닙니다.")


if __name__ == "__main__":
    main()
