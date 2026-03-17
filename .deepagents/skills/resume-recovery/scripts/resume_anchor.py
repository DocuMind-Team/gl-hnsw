from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: resume_anchor.py <manifest.json>", file=sys.stderr)
        return 2
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    completed = set(payload.get("completed_stages", []))
    ordered = ["dossiers", "candidates", "judgments", "checks", "reviews", "memory"]
    next_stage = next((stage for stage in ordered if stage not in completed), None)
    print(json.dumps({"next_stage": next_stage or "", "completed_stages": sorted(completed)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
