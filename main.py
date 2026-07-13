"""QLD/TQQQ 전략 백테스트 실행 스크립트."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
import pandas as pd

import data
import strategies as st
import backtest as bt

pd.options.display.float_format = "{:.3f}".format


def main():
    df = data.build_dataset()
    print(f"데이터 구간: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)}일)\n")

    print("=== 합성 시리즈 검증 (실제 ETF 대비) ===")
    print(data.validate_synthetic(df), "\n")

    strategy_map = {
        "QLD 보유": lambda d: st.buy_and_hold(d, "QLD"),
        "TQQQ 보유": lambda d: st.buy_and_hold(d, "TQQQ"),
        "QLD 50/TQQQ 50": lambda d: st.fixed_mix(d, 0.5, 0.5),
        "TQQQ 200일선": lambda d: st.sma_timing(d, "TQQQ", 200),
        "QLD 200일선": lambda d: st.sma_timing(d, "QLD", 200),
        "TQQQ 변동성타겟 35%": lambda d: st.vol_target(d, "TQQQ", 0.35),
        "레짐 로테이션": st.regime_rotation,
    }

    curves, table = bt.compare(df, strategy_map)

    print("=== 전략 비교 (1999-03 ~ 현재, 합성 시리즈, 거래비용 5bp) ===")
    print(table.sort_values("CAGR", ascending=False))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    for name in curves:
        ax1.plot(curves.index, curves[name], label=name, linewidth=1)
    ax1.set_yscale("log")
    ax1.set_title("Cumulative Growth (log scale)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    for name in curves:
        dd = curves[name] / curves[name].cummax() - 1
        ax2.plot(dd.index, dd, label=name, linewidth=0.8)
    ax2.set_title("Drawdown")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("results.png", dpi=120)
    print("\n차트 저장: results.png")

    table.to_csv("results.csv")


if __name__ == "__main__":
    main()
