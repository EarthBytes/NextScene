import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build Faiss vector index")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/faiss/items.index"),
        help="Path to save the Faiss index",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Faiss index build at {args.output} — implemented in Step 8.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
