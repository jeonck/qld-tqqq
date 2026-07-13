"""일일 신호를 계산해 docs/ 아래 게시판(정적 HTML)을 생성한다.

- docs/history.json : 날짜별 신호 기록 (append, 같은 날짜는 갱신)
- docs/index.html   : 최신 신호 카드 + 과거 기록 테이블
GitHub Actions가 장 마감 후 실행하고 커밋하면 GitHub Pages로 서빙된다.
"""
import json
import os
from pathlib import Path

import data
import signal_now

DOCS = Path(__file__).parent / "docs"
HISTORY = DOCS / "history.json"

STATE_LABEL = {0: "기본 보유", 1: "TQQQ 진입 중", 2: "손절 후 현금 대기"}
STATE_BADGE = {0: "base", 1: "tqqq", 2: "stopped"}
STATE_ALLOC = {0: "QQQM 30% / QLD 70%",
               1: "QQQM 30% / QLD 40% / TQQQ 30%",
               2: "QQQM 30% / 현금 70%"}


def build_record(s: dict) -> dict:
    p = signal_now.PARAMS
    rec = {
        "date": str(s["date"].date()),
        "state": s["state"],
        "state_label": STATE_LABEL[s["state"]],
        "alloc": STATE_ALLOC[s["state"]],
        "qqq": round(float(s["qqq"]), 2),
        "sma200": round(float(s["sma"]), 2),
        "above_sma": bool(s["above_sma"]),
        "high52": round(float(s["high52"]), 2),
        "dd": round(float(s["dd"]), 4),
    }
    if s["state"] == 0:
        rec["note"] = (f"진입 트리거: QQQ {s['high52'] * (1 + p['dip_entry']):,.2f} 이하"
                       if s["above_sma"] else "200일선 아래 — 진입 대기")
    elif s["state"] == 1:
        dd_tq = s["tqqq_px"] / s["tqqq_peak"] - 1
        rec["note"] = (f"익절 목표 QQQ {s['target_high']:,.2f} / "
                       f"손절 여유 {dd_tq - p['trail_stop']:+.1%}p")
    else:
        rec["note"] = f"복귀 조건: QQQ {s['target_high']:,.2f} 회복"
    return rec


def load_history() -> list:
    if HISTORY.exists():
        return json.loads(HISTORY.read_text())
    return []


def render_html(history: list) -> str:
    latest = history[0]
    badge = STATE_BADGE[latest["state"]]
    rows = "\n".join(
        f"""<tr>
        <td>{r['date']}</td>
        <td><span class="badge {STATE_BADGE[r['state']]}">{r['state_label']}</span></td>
        <td>{r['alloc']}</td>
        <td class="num">{r['qqq']:,.2f}</td>
        <td class="num">{r['dd']:+.1%}</td>
        <td class="num">{'▲' if r['above_sma'] else '▼'} {r['sma200']:,.2f}</td>
        <td class="note">{r['note']}</td>
        </tr>"""
        for r in history)

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QLD/TQQQ 일일 신호</title>
<style>
:root {{ --bg:#fff; --fg:#1a1a2e; --muted:#667; --card:#f5f6fa; --line:#e2e4ec;
        --green:#0a7d38; --red:#c0392b; --amber:#b7791f; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#12141c; --fg:#e8eaf2; --muted:#9aa; --card:#1c1f2b; --line:#2a2e3e;
          --green:#4ade80; --red:#f87171; --amber:#fbbf24; }} }}
* {{ box-sizing:border-box; margin:0; }}
body {{ background:var(--bg); color:var(--fg);
       font:15px/1.6 -apple-system,'Apple SD Gothic Neo','Noto Sans KR',sans-serif;
       max-width:960px; margin:0 auto; padding:2rem 1rem; }}
h1 {{ font-size:1.4rem; margin-bottom:.3rem; }}
.sub {{ color:var(--muted); font-size:.85rem; margin-bottom:1.5rem; }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
        padding:1.2rem 1.5rem; margin-bottom:2rem; }}
.card .state {{ font-size:1.5rem; font-weight:700; margin:.2rem 0 .6rem; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:.8rem; }}
.grid div span {{ display:block; color:var(--muted); font-size:.75rem; }}
.grid div b {{ font-size:1.05rem; }}
table {{ width:100%; border-collapse:collapse; font-size:.85rem; }}
th,td {{ padding:.5rem .6rem; text-align:left; border-bottom:1px solid var(--line);
        white-space:nowrap; }}
td.num {{ font-variant-numeric:tabular-nums; }}
td.note {{ color:var(--muted); white-space:normal; }}
.badge {{ padding:.15rem .55rem; border-radius:999px; font-size:.78rem; font-weight:600; }}
.badge.base {{ background:color-mix(in srgb,var(--green) 15%,transparent); color:var(--green); }}
.badge.tqqq {{ background:color-mix(in srgb,var(--red) 15%,transparent); color:var(--red); }}
.badge.stopped {{ background:color-mix(in srgb,var(--amber) 18%,transparent); color:var(--amber); }}
.wrap {{ overflow-x:auto; }}
.foot {{ color:var(--muted); font-size:.75rem; margin-top:1.5rem; }}
</style>
</head>
<body>
<h1>QLD/TQQQ 일일 신호 게시판</h1>
<p class="sub">규칙: QQQM 30/QLD 70 기본 보유 → QQQ 52주 고점 대비 -10% & 200일선 위에서
QLD 30%p를 TQQQ로 전환(30/40/30) → 전고점 회복 시 익절 / TQQQ -30% 트레일링 손절 시
QLD 몫 현금 대피 · 매 거래일 장 마감 후 자동 갱신</p>

<div class="card">
  <span class="badge {badge}">{latest['date']}</span>
  <div class="state">{latest['state_label']} — {latest['alloc']}</div>
  <div class="grid">
    <div><span>QQQ 종가</span><b>{latest['qqq']:,.2f}</b></div>
    <div><span>52주 고점 대비</span><b>{latest['dd']:+.1%}</b></div>
    <div><span>200일선 ({'위' if latest['above_sma'] else '아래'})</span><b>{latest['sma200']:,.2f}</b></div>
    <div><span>다음 액션</span><b style="font-size:.9rem">{latest['note']}</b></div>
  </div>
</div>

<div class="wrap">
<table>
<thead><tr><th>날짜</th><th>상태</th><th>목표 배분</th><th>QQQ</th><th>낙폭</th>
<th>200일선</th><th>비고</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
</div>

<p class="foot">백테스트(1999–2026, CAGR 16.0% / MDD -56%) 기반 리서치 도구이며 투자자문이 아닙니다.
데이터: Yahoo Finance 수정종가. <a href="https://github.com/jeonck/qld-tqqq">GitHub</a></p>
</body>
</html>"""


def main():
    DOCS.mkdir(exist_ok=True)
    df = data.build_dataset(force=os.environ.get("CI") == "true")
    s = signal_now.current_state(df)
    rec = build_record(s)

    history = [r for r in load_history() if r["date"] != rec["date"]]
    history.insert(0, rec)
    history.sort(key=lambda r: r["date"], reverse=True)
    HISTORY.write_text(json.dumps(history, ensure_ascii=False, indent=1))

    (DOCS / "index.html").write_text(render_html(history))
    print(f"발행 완료: {rec['date']} [{rec['state_label']}] {rec['alloc']}")


if __name__ == "__main__":
    main()
