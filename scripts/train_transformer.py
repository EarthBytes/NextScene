"""Train the generative sequence transformer on interaction histories."""

import os

# Must be set before torch/faiss import OpenMP (macOS PyTorch + faiss-cpu conflict).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import app.ml_runtime  # noqa: F401

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.config import settings
from app.db.session import SessionLocal
from app.models_ml.sequence_transformer import SequenceTransformerConfig
from app.services.clip_embeddings import resolve_device
from app.services.sequence_training import TrainingConfig, run_training


def check_database(session) -> None:
    try:
        session.execute(text("SELECT 1"))
    except OperationalError:
        print("Cannot connect to PostgreSQL on localhost:5432.")
        print("Start Docker Desktop, then run: docker compose up -d postgres")
        raise SystemExit(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the next-item sequence transformer")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(settings.transformer_model_path),
        help="Directory for checkpoints and config (default: models/transformer)",
    )
    parser.add_argument(
        "--sequences-cache",
        type=Path,
        default=Path(settings.sequences_cache_path),
        help="Directory with pre-built user sequences (default: data/sequences)",
    )
    parser.add_argument(
        "--rebuild-sequences",
        action="store_true",
        help="Rebuild the sequence cache from PostgreSQL before training",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=50)
    parser.add_argument(
        "--negatives",
        type=int,
        default=64,
        help="Total random+hard negatives per batch (split evenly by default)",
    )
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience on val Recall@10",
    )
    parser.add_argument("--max-users", type=int, default=None, help="Train on a user subset (e.g. 10000)")
    parser.add_argument(
        "--windows-per-user",
        type=int,
        default=5,
        help="Random training windows sampled per user each epoch",
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=4,
        help="Number of user shards streamed per epoch",
    )
    parser.add_argument(
        "--min-rating",
        type=float,
        default=3.5,
        help="Keep rating interactions at or above this value",
    )
    parser.add_argument(
        "--no-rating-filter",
        action="store_true",
        help="Include all rating interactions regardless of score",
    )
    parser.add_argument(
        "--no-hard-negatives",
        action="store_true",
        help="Disable in-catalog hard negatives (random catalog negatives only)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default=None, help="Torch device (default: cuda / mps / cpu)")
    parser.add_argument(
        "--user-batches",
        type=int,
        default=1,
        help="Split train users into N sequential batches (default: 1 = all users)",
    )
    parser.add_argument(
        "--user-batch",
        type=int,
        default=None,
        help="Run only batch N (0-indexed). Default: run all batches sequentially.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume model weights from a checkpoint path (for batch 0 or single run)",
    )
    return parser.parse_args()


def main() -> int:
    try:
        import torch  # noqa: F401
    except ImportError:
        print("PyTorch is required. Install with: pip install -r requirements-ml.txt")
        return 1

    args = parse_args()
    resolved_device = resolve_device(args.device)
    min_rating = None if args.no_rating_filter else args.min_rating
    negative_split = max(args.negatives // 2, 1)

    model_config = SequenceTransformerConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        clip_model=settings.clip_model_name,
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        negatives_per_sample=args.negatives,
        random_negatives_per_sample=negative_split,
        hard_negatives_per_sample=negative_split,
        temperature=args.temperature,
        early_stopping_patience=args.patience,
        num_workers=args.num_workers,
        seed=args.seed,
        min_rating=min_rating,
        max_users=args.max_users,
        sequences_cache_dir=str(args.sequences_cache),
        rebuild_sequences=args.rebuild_sequences,
        windows_per_user=args.windows_per_user,
        num_shards=args.shards,
        use_hard_negatives=not args.no_hard_negatives,
        user_batches=args.user_batches,
        user_batch=args.user_batch,
        resume_checkpoint=args.resume,
    )

    session = SessionLocal()
    try:
        check_database(session)
        epochs_note = (
            f"{max(1, args.epochs // args.user_batches)} epochs/batch × {args.user_batches} batches"
            if args.user_batches > 1
            else f"{args.epochs} epochs"
        )
        print(
            f"Training sequence transformer "
            f"(device={resolved_device}, {epochs_note}, "
            f"batch_size={args.batch_size}, max_users={args.max_users or 'all'}, "
            f"windows_per_user={args.windows_per_user}, shards={args.shards}, "
            f"hard_negatives={not args.no_hard_negatives}) ..."
        )
        result = run_training(
            session,
            output_dir=args.output,
            model_config=model_config,
            training_config=training_config,
            device=args.device,
        )
    finally:
        session.close()

    for key, value in result.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
