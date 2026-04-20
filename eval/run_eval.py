"""Retrieval evaluation harness.

Reads eval/golden.jsonl, runs retrieve() for each query, computes:
  - source_recall: fraction of expected_sources present in top-k
  - modality_coverage: fraction of expected_modalities present in top-k
  - avg chunks retrieved

Does NOT call the LLM — pure retrieval metric. Ollama not required.

Usage:
  python eval/run_eval.py
  python eval/run_eval.py --hyde
  python eval/run_eval.py --top-k 10
"""
import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from logging_config import setup_logging  # noqa: E402
from retrieval.retriever import retrieve  # noqa: E402
from vectorstore.store import get_count  # noqa: E402


def _basename(p):
    return os.path.basename(p or "")


def _load_golden(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip() and not line.startswith("//")]


def _evaluate_case(case, use_hyde):
    chunks = retrieve(case["query"], use_hyde=use_hyde)

    got_sources = {_basename(m.get("source", "")) for _, m in chunks}
    got_modalities = {m.get("modality") for _, m in chunks}

    exp_sources = set(case.get("expected_sources", []))
    exp_mods = set(case.get("expected_modalities", []))

    src_recall = (
        len(exp_sources & got_sources) / len(exp_sources) if exp_sources else 1.0
    )
    mod_cov = (
        len(exp_mods & got_modalities) / len(exp_mods) if exp_mods else 1.0
    )

    return {
        "query": case["query"],
        "retrieved_chunks": len(chunks),
        "got_sources": sorted(got_sources),
        "source_recall": round(src_recall, 3),
        "modality_coverage": round(mod_cov, 3),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=os.path.join(ROOT, "eval", "golden.jsonl"))
    parser.add_argument("--hyde", action="store_true")
    args = parser.parse_args()

    setup_logging()
    if get_count() == 0:
        print("Collection empty. Ingest data first.")
        sys.exit(1)

    cases = _load_golden(args.golden)
    print(f"Running {len(cases)} eval cases against {get_count()} indexed chunks "
          f"(hyde={args.hyde})\n")

    results = [_evaluate_case(c, args.hyde) for c in cases]

    total_recall = sum(r["source_recall"] for r in results) / len(results)
    total_cov = sum(r["modality_coverage"] for r in results) / len(results)
    avg_chunks = sum(r["retrieved_chunks"] for r in results) / len(results)

    for r in results:
        print(f"  Q: {r['query'][:60]}")
        print(f"     chunks={r['retrieved_chunks']} recall={r['source_recall']} "
              f"modcov={r['modality_coverage']} sources={r['got_sources']}")

    print()
    print(f"Mean source_recall   : {total_recall:.3f}")
    print(f"Mean modality_coverage: {total_cov:.3f}")
    print(f"Mean retrieved chunks : {avg_chunks:.1f}")


if __name__ == "__main__":
    main()
