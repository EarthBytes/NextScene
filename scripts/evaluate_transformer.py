"""Offline evaluation for the trained sequence transformer."""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import app.ml_runtime  # noqa: F401

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.config import settings
from app.db.session import SessionLocal
from app.services.clip_embeddings import resolve_device
from app.services.sequence_evaluation import (
    BEST_FILENAME,
    EvalConfig,
    run_offline_evaluation,
    write_eval_report,
)


def check_database(session) -> None:
    try:
        session.execute(text("SELECT 1"))
    except OperationalError:
        print("Cannot connect to PostgreSQL on localhost:5432.")
        print("Start Docker Desktop, then run: docker compose up -d postgres")
        raise SystemExit(1)


def parse_k_values(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("At least one k value is required")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained sequence transformer")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(settings.transformer_model_path) / BEST_FILENAME,
        help="Checkpoint path (default: models/transformer/best.pt)",
    )
    parser.add_argument(
        "--sequences-cache",
        type=Path,
        default=Path(settings.sequences_cache_path),
        help="Cached user sequences (default: data/sequences)",
    )
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--k", type=parse_k_values, default=parse_k_values("10,20,50"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=None, help="Torch device (default: cuda / mps / cpu)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON report path (default: reports/eval_<timestamp>.json)",
    )
    return parser.parse_args()


def print_report(report: dict) -> None:
    metadata = report["metadata"]
    print(
        f"Evaluated {metadata['samples']:,} {metadata['split']} samples "
        f"(trained on {metadata['training_users']:,} users)"
    )
    print(f"Checkpoint: {metadata['checkpoint']}")
    print("")
    for model_name, metrics in report["models"].items():
        metric_parts = "  ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()))
        print(f"{model_name:>22}  {metric_parts}")
    print("")
    if metadata["beats_avg_embedding_baseline"]:
        print("Transformer beats avg-history-embedding baseline on Recall@10.")
    else:
        print("Transformer does NOT beat avg-history-embedding baseline on Recall@10.")


def main() -> int:
    try:
        import torch  # noqa: F401
    except ImportError:
        print("PyTorch is required. Install with: pip install -r requirements-ml.txt")
        return 1

    args = parse_args()
    if not args.checkpoint.is_file():
        print(f"Checkpoint not found: {args.checkpoint}")
        return 1
    if not args.sequences_cache.joinpath("sequences.npz").is_file():
        print(f"Sequence cache not found: {args.sequences_cache}")
        print("Run: python scripts/build_training_sequences.py --output data/sequences")
        return 1

    output_path = args.output
    if output_path is None:
        stamp = datetime.now(UTC).strftime("%Y-%m-%d")
        output_path = Path("reports") / f"eval_{stamp}.json"

    eval_config = EvalConfig(
        checkpoint_path=args.checkpoint,
        sequences_cache_dir=args.sequences_cache,
        split=args.split,
        k_values=args.k,
        batch_size=args.batch_size,
        device=args.device,
    )

    session = SessionLocal()
    try:
        check_database(session)
        print(
            f"Running offline evaluation "
            f"(device={resolve_device(args.device)}, split={args.split}, k={args.k}) ..."
        )
        report = run_offline_evaluation(session, eval_config)
    finally:
        session.close()

    write_eval_report(report, output_path)
    print_report(report)
    print(f"Report saved to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
