import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Ingest MovieLens 20M into the database")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/movielens"),
        help="Directory containing ratings.csv, movies.csv, tags.csv",
    )
    args = parser.parse_args()

    required = ["ratings.csv", "movies.csv", "tags.csv"]
    missing = [f for f in required if not (args.data_dir / f).exists()]
    if missing:
        print(f"Missing files in {args.data_dir}: {', '.join(missing)}")
        print("Download from https://grouplens.org/datasets/movielens/20m/")
        return 1

    print("MovieLens files found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
