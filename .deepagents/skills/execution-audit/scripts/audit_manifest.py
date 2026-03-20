from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: audit_manifest.py <manifest.json>", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    payload = json.loads(path.read_text(encoding="utf-8"))
    completed = payload.get("completed_stages", [])
    retries = payload.get("retry_counts", {})
    print(
        json.dumps(
            {
                "anchor_doc_id": payload.get("anchor_doc_id"),
                "completed_stage_count": len(completed),
                "completed_stages": completed,
                "retry_counts": retries,
                "halt_requested": bool(payload.get("halt_requested")),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
