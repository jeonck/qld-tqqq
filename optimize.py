"""시나리오 그리드 서치 실행 및 최적 배분 리포트."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
import pandas as pd

import data
import backtest as bt
import scenarios as sc
import strategies as st

pd.options.display.float_format = "{:.3f}".format
pd.options.display.width = 200


def main():
    df = data.build_dataset()
    print(f"데이터: {df.index[0].date()} ~ {df.index[-1].date()}, "
          f"시나리오 그리드 서치 시작...\n")

    res = sc.grid_search(df)
    res.to_csv("grid_results.csv", index=False)
    print(f"총 {len(res)}개 시나리오 완료 → grid_results.csv\n")

    cagr_cols = [c for c in res.columns if c.startswith("CAGR ")]

    print("=== CAGR(전기간) 상위 10 ===")
    print(res.sort_values("CAGR 전기간", ascending=False).head(10).to_string(index=False))

    print("\n=== Calmar(CAGR/MDD) 상위 10 — 위험조정 기준 ===")
    print(res.sort_values("Calmar", ascending=False).head(10).to_string(index=False))

    # 견고성: 세 구간 모두 상위 40% 이내인 조합
    robust = res.copy()
    for c in cagr_cols[1:]:
        robust = robust[robust[c] >= res[c].quantile(0.6)]
    print(f"\n=== 견고한 조합 (모든 구간 상위 40%): {len(robust)}개, Calmar 상위 10 ===")
    print(robust.sort_values("Calmar", ascending=False).head(10).to_string(index=False))

    # 벤치마크 + 대표 전략 곡선
    best_cagr = res.sort_values("CAGR 전기간", ascending=False).iloc[0]
    best_robust = robust.sort_values("Calmar", ascending=False).iloc[0]

    def curve_of(p):
        w = sc.escalation_weights(df, p["QQQM비중"], p["진입낙폭"], p["TQQQ비중"],
                                  bool(p["선하진입"]),
                                  None if p["손절"] == 0 else p["손절"],
                                  bool(p["방어"]))
        return bt.run(df, w)

    curves = pd.DataFrame({
        "최대수익 조합": curve_of(best_cagr),
        "견고한 최적 조합": curve_of(best_robust),
        "QQQ 보유": bt.run(df, _qqq_hold(df)),
        "QLD 보유": bt.run(df, st.buy_and_hold(df, "QLD")),
        "QQQM50/QLD50 고정": bt.run(df, _fixed(df, 0.5)),
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    for name in curves:
        ax1.plot(curves.index, curves[name], label=name, linewidth=1)
    ax1.set_yscale("log"); ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
    ax1.set_title("누적 성과 (로그)")
    for name in curves:
        dd = curves[name] / curves[name].cummax() - 1
        ax2.plot(dd.index, dd, label=name, linewidth=0.8)
    ax2.set_title("낙폭"); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("optimal.png", dpi=120)
    print("\n차트 저장: optimal.png")

    print("\n[최대수익 조합]"); print(best_cagr.to_string())
    print("\n[견고한 최적 조합]"); print(best_robust.to_string())


def _qqq_hold(df):
    w = pd.DataFrame(0.0, index=df.index, columns=["QQQ", "QLD", "TQQQ", "cash"])
    w["QQQ"] = 1.0
    return w.shift(1).fillna(0.0)


def _fixed(df, q):
    w = pd.DataFrame(0.0, index=df.index, columns=["QQQ", "QLD", "TQQQ", "cash"])
    w["QQQ"], w["QLD"] = q, 1 - q
    return w.shift(1).fillna(0.0)


if __name__ == "__main__":
    main()
