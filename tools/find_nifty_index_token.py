import csv
from pathlib import Path


def main() -> int:
    p = Path("scripmaster.csv")
    if not p.exists():
        print("scripmaster.csv not found in current directory")
        return 2

    targets = {"NIFTY", "NIFTY 50", "NIFTY50"}
    matches = []
    fuzzy = []

    with p.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            exch = (row.get("exchange") or "").strip().upper()
            if exch != "NSE":
                continue

            ts = (row.get("tradingsymbol") or "").strip().upper()
            name = (row.get("name") or "").strip()
            inst = (row.get("instrument_type") or "").strip().upper()
            seg = (row.get("segment") or "").strip().upper()

            if ts in targets or ts == "NIFTY":
                matches.append((row.get("instrument_token"), ts, name, inst, seg))
            # Some masters encode as NIFTY with instrument_type INDEX/INDICES
            elif ts.startswith("NIFTY") and inst in {"INDEX", "INDICES"}:
                matches.append((row.get("instrument_token"), ts, name, inst, seg))

            # Fuzzy capture: anything NSE with NIFTY in symbol/name.
            if ("NIFTY" in ts) or ("NIFTY" in name.upper()):
                fuzzy.append((row.get("instrument_token"), ts, name, inst, seg))

            if len(matches) >= 25 and len(fuzzy) >= 50:
                break

    print(f"Found {len(matches)} NSE matches")
    for m in matches:
        print(m)

    print("\nFuzzy NSE rows containing NIFTY (first 50):")
    for m in fuzzy[:50]:
        print(m)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
