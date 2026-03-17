from __future__ import annotations

import json
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: check_task_budget.py <delegation_round> <task_iteration_cap>", file=sys.stderr)
        return 2
    round_count = int(sys.argv[1])
    cap = int(sys.argv[2])
    print(json.dumps({"delegation_round": round_count, "task_iteration_cap": cap, "within_budget": round_count < cap}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
