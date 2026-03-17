from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: score_anchor_utility.py <review_bundle.json>", file=sys.stderr)
        return 2
    reviews = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")).get("reviews", [])
    utilities = [float(item.get("reviewed_utility_score", 0.0) or 0.0) for item in reviews]
    kept = [item for item in reviews if item.get("keep")]
    print(
        json.dumps(
            {
                "review_count": len(reviews),
                "kept_count": len(kept),
                "top_reviewed_utility": max(utilities) if utilities else 0.0,
                "mean_reviewed_utility": (sum(utilities) / len(utilities)) if utilities else 0.0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
