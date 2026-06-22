#!/usr/bin/env python3
"""generate_daily_paper_report.py — aggregate paper forward-test journals into a daily report."""
import argparse, csv, json, statistics
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS = REPO_ROOT / "reports"
PAPER_FWD = REPORTS / "paper_forward_test"

def find_latest_dir() -> Path | None:
    if not PAPER_FWD.exists():
        return None
    dirs = sorted(PAPER_FWD.glob("paper_decisions_*/"), key=lambda p: p.stat().st_mtime, reverse=True)
    # Actually decisions are files, not dirs
    return None

def collect_journal_entries() -> list[dict]:
    """Collect all JSONL entries from latest forward-test run."""
    entries = []
    if not PAPER_FWD.exists():
        return entries

    jsonl_files = sorted(PAPER_FWD.glob("paper_decisions_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not jsonl_files:
        return entries

    latest = jsonl_files[0]
    with open(latest, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    return entries

def generate_daily_report(output_date: str = datetime.now(timezone.utc).strftime("%Y%m%d")) -> dict:
    entries = collect_journal_entries()

    n = len(entries)
    skip_count = sum(1 for e in entries if e.get("final_action") == "SKIP")
    trade_count = sum(1 for e in entries if e.get("final_action") == "TRADE")

    probabilities = [e.get("probability", 0.0) for e in entries]

    skip_reasons: dict[str, int] = {}
    for e in entries:
        if e.get("final_action") == "SKIP":
            for r in e.get("blocking_reasons", []):
                skip_reasons[r] = skip_reasons.get(r, 0) + 1

    spreads = [e.get("spread_pct", 0.0) for e in entries if e.get("spread_pct", 0) > 0]
    premiums = [e.get("premium", 0.0) for e in entries if e.get("premium", 0) > 0]

    prob_mean = statistics.mean(probabilities) if probabilities else 0.0

    # Alerts
    alerts = []
    if n == 0:
        alerts.append("NO_DECISIONS")
    if trade_count == 0 and skip_count > 0:
        alerts.append("ALL_SKIP")
    if prob_mean < 0.4:
        alerts.append("LOW_SIGNAL")
    if prob_mean < 0.45:
        alerts.append("BELOW_THRESHOLD")

    summary = {
        "report_date": output_date,
        "total_decisions": n,
        "total_skip": skip_count,
        "total_trade": trade_count,
        "skip_reasons": skip_reasons,
        "probability_mean": round(prob_mean, 4),
        "probability_min": round(min(probabilities), 4) if probabilities else 0.0,
        "probability_max": round(max(probabilities), 4) if probabilities else 0.0,
        "spread_mean_pct": round(statistics.mean(spreads) * 100, 4) if spreads else 0.0,
        "premium_mean": round(statistics.mean(premiums), 2) if premiums else 0.0,
        "alerts": alerts,
        "status": "REVIEW_NEEDED" if alerts else "OK",
    }

    # Write JSON
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    json_path = REPORTS / f"daily_paper_report_{output_date}_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Write Markdown
    md_lines = [
        f"# Daily Paper Report — {output_date}",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Summary",
        f"- Decisions: {n} | SKIP: {skip_count} | TRADE: {trade_count}",
        f"- Avg probability: {prob_mean:.4f}",
        f"- Alerts: {', '.join(alerts) if alerts else 'None'}",
        "",
        "## Skip Reasons",
    ]
    for reason, count in sorted(skip_reasons.items(), key=lambda x: -x[1]):
        md_lines.append(f"- `{reason}`: {count}")

    md_lines.append("")
    md_lines.append("## Alerts")
    if alerts:
        md_lines.extend(f"- **{a}**" for a in alerts)
    else:
        md_lines.append("- None")

    md_path = REPORTS / f"daily_paper_report_{output_date}_{ts}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return summary

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    args = parser.parse_args()
    result = generate_daily_report(args.date)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()