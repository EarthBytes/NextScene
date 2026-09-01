"""Check that recommended items never overlap with a user's interaction history."""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

BACKEND_ROOT = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import app.ml_runtime  # noqa: F401

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.config import settings
from app.db.session import SessionLocal
from app.services.clip_embeddings import resolve_device
from app.services.recommendation_service import (
    Recommendation,
    load_recommendation_service,
    load_user_seen_items,
    popularity_recommendations,
    try_load_serving_context,
)


@dataclass(frozen=True)
class OverlapResult:
    user_id: int
    history_count: int
    recommendation_count: int
    overlap_item_ids: tuple[int, ...]
    model_version: str

    @property
    def passed(self) -> bool:
        return not self.overlap_item_ids


def check_database(session) -> None:
    try:
        session.execute(text("SELECT 1"))
    except OperationalError:
        print("Cannot connect to PostgreSQL on localhost:5432.")
        print("Start Docker Desktop, then run: docker compose up -d postgres")
        raise SystemExit(1)


def parse_user_ids(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def sample_user_ids(session, *, count: int, seed: int, min_interactions: int) -> list[int]:
    rows = session.execute(
        text(
            """
            SELECT user_id
            FROM (
                SELECT user_id, COUNT(DISTINCT item_id) AS item_count
                FROM interactions
                GROUP BY user_id
                HAVING COUNT(DISTINCT item_id) >= :min_interactions
            ) users
            ORDER BY md5(user_id::text || :seed)
            LIMIT :count
            """
        ),
        {"count": count, "seed": str(seed), "min_interactions": min_interactions},
    )
    return [int(row.user_id) for row in rows]


def fetch_recommendations_api(api_url: str, user_id: int, k: int) -> tuple[list[Recommendation], str]:
    query = urlencode({"user_id": user_id, "k": k})
    url = f"{api_url.rstrip('/')}/api/recommendations?{query}"
    try:
        with urlopen(url, timeout=60) as response:
            payload = json.load(response)
    except URLError as exc:
        raise RuntimeError(f"Failed to fetch recommendations for user {user_id}: {exc}") from exc

    model_version = str(payload.get("model_version", "unknown"))
    recommendations = [
        Recommendation(
            item_id=int(rec["item_id"]),
            title=rec.get("title"),
            score=float(rec["score"]),
        )
        for rec in payload.get("recommendations", [])
    ]
    return recommendations, model_version


def check_user_overlap(
    session,
    user_id: int,
    *,
    k: int,
    service=None,
    popularity_ranking: list[int] | None = None,
    model_version: str = "unknown",
    api_url: str | None = None,
) -> OverlapResult:
    seen_items = load_user_seen_items(session, user_id)

    if api_url is not None:
        recommendations, model_version = fetch_recommendations_api(api_url, user_id, k)
    elif service is not None:
        recommendations = service.recommend(session, user_id=user_id, k=k)
        model_version = service.model_version
    else:
        if popularity_ranking is None:
            raise ValueError("popularity_ranking is required for popularity fallback checks")
        recommendations = popularity_recommendations(
            session,
            user_id=user_id,
            k=k,
            popularity_ranking=popularity_ranking,
        )
        model_version = model_version

    recommended_ids = {rec.item_id for rec in recommendations}
    overlap = tuple(sorted(seen_items & recommended_ids))
    return OverlapResult(
        user_id=user_id,
        history_count=len(seen_items),
        recommendation_count=len(recommendations),
        overlap_item_ids=overlap,
        model_version=model_version,
    )


def load_titles(session, item_ids: list[int]) -> dict[int, str | None]:
    if not item_ids:
        return {}
    rows = session.execute(
        text("SELECT item_id, title FROM items WHERE item_id = ANY(:item_ids)"),
        {"item_ids": item_ids},
    )
    return {int(row.item_id): row.title for row in rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify recommendations do not include items the user has already seen",
    )
    parser.add_argument("--users", type=int, default=200, help="Number of users to sample (default: 200)")
    parser.add_argument("--k", type=int, default=50, help="Recommendations per user (default: 50)")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed (default: 42)")
    parser.add_argument(
        "--min-interactions",
        type=int,
        default=3,
        help="Only sample users with at least this many distinct items (default: 3)",
    )
    parser.add_argument(
        "--user-ids",
        type=parse_user_ids,
        default=None,
        help="Comma-separated user ids to check instead of sampling",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="Use a running API instead of loading the model locally (e.g. http://localhost:8000)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(settings.transformer_model_path),
        help="Model directory for direct serving mode",
    )
    parser.add_argument("--device", default=None, help="Torch device for direct serving mode")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path (default: reports/overlap_<timestamp>.json)",
    )
    parser.add_argument(
        "--show-failures",
        type=int,
        default=20,
        help="Max failing users to print in detail (default: 20)",
    )
    return parser.parse_args()


def print_summary(results: list[OverlapResult], titles: dict[int, str | None], show_failures: int) -> None:
    failures = [result for result in results if not result.passed]
    empty_recs = [result for result in results if result.recommendation_count == 0]

    print(f"Checked {len(results)} users")
    print(f"Passed: {len(results) - len(failures)}")
    print(f"Failed (overlap): {len(failures)}")
    print(f"Empty recommendations: {len(empty_recs)}")

    if failures:
        print("\nFailures:")
        for result in failures[:show_failures]:
            overlap_labels = [
                f"{item_id} ({titles.get(item_id) or 'unknown title'})"
                for item_id in result.overlap_item_ids
            ]
            print(
                f"  user {result.user_id}: overlap={overlap_labels} "
                f"(history={result.history_count}, recs={result.recommendation_count}, "
                f"model={result.model_version})"
            )
        if len(failures) > show_failures:
            print(f"  ... and {len(failures) - show_failures} more")


def build_report(results: list[OverlapResult], args: argparse.Namespace) -> dict:
    failures = [result for result in results if not result.passed]
    return {
        "metadata": {
            "checked_at": datetime.now(UTC).isoformat(),
            "users_checked": len(results),
            "k": args.k,
            "seed": args.seed,
            "min_interactions": args.min_interactions,
            "api_url": args.api_url,
            "model_dir": None if args.api_url else str(args.model_dir),
            "failures": len(failures),
            "empty_recommendations": sum(1 for result in results if result.recommendation_count == 0),
        },
        "failures": [asdict(result) for result in failures],
        "results": [asdict(result) for result in results],
    }


def main() -> int:
    args = parse_args()
    session = SessionLocal()
    check_database(session)

    user_ids = args.user_ids or sample_user_ids(
        session,
        count=args.users,
        seed=args.seed,
        min_interactions=args.min_interactions,
    )
    if not user_ids:
        print("No users matched the sampling criteria.")
        return 1

    service = None
    popularity_ranking = None
    model_version = "unknown"
    if args.api_url is None:
        if args.model_dir.is_dir() and (args.model_dir / "best.pt").is_file():
            device = resolve_device(args.device)
            print(f"Loading model from {args.model_dir} on {device}...")
            service = load_recommendation_service(
                session,
                model_dir=args.model_dir,
                inference_device=device,
            )
            model_version = service.model_version
        else:
            print("Model checkpoint not found; using popularity fallback.")
            serving = try_load_serving_context(session)
            popularity_ranking = serving.popularity_ranking
            model_version = serving.model_version

    print(
        f"Checking {len(user_ids)} users "
        f"(k={args.k}, mode={'api' if args.api_url else 'direct'})..."
    )

    results: list[OverlapResult] = []
    for index, user_id in enumerate(user_ids, start=1):
        result = check_user_overlap(
            session,
            user_id,
            k=args.k,
            service=service,
            popularity_ranking=popularity_ranking,
            model_version=model_version,
            api_url=args.api_url,
        )
        results.append(result)
        if index % 25 == 0 or index == len(user_ids):
            print(f"  processed {index}/{len(user_ids)}")

    overlap_item_ids = sorted({item_id for result in results for item_id in result.overlap_item_ids})
    titles = load_titles(session, overlap_item_ids)
    print_summary(results, titles, show_failures=args.show_failures)

    report = build_report(results, args)
    output_path = args.output
    if output_path is None:
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        output_path = reports_dir / f"overlap_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"\nWrote report to {output_path}")

    session.close()
    return 1 if any(not result.passed for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
