from __future__ import annotations

import argparse
from pathlib import Path

from hnsw_logic.evaluation.beir import evaluate_beir_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate gl-hnsw retrieval transfer on a BEIR dataset.")
    parser.add_argument("--dataset", default="scifact", help="BEIR dataset name, e.g. scifact, arguana, nfcorpus")
    parser.add_argument("--split", default="test", help="BEIR split name")
    parser.add_argument("--query-limit", type=int, default=None, help="Optional limit on the number of queries")
    parser.add_argument("--corpus-limit", type=int, default=None, help="Optional limit on the number of documents; keeps all positives and fills with hard negatives")
    parser.add_argument("--cache-root", type=Path, default=None, help="Optional cache directory for downloaded BEIR zips")
    parser.add_argument("--work-root", type=Path, default=None, help="Optional isolated work directory")
    parser.add_argument("--allow-stub", action="store_true", help="Allow stub-provider runs for local debugging only")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    report = evaluate_beir_dataset(
        repo_root=repo_root,
        dataset=args.dataset,
        split=args.split,
        query_limit=args.query_limit,
        corpus_limit=args.corpus_limit,
        cache_root=args.cache_root,
        work_root=args.work_root,
        allow_stub=args.allow_stub,
    )
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
