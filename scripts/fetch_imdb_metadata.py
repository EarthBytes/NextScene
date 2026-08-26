import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Fetch IMDb metadata via OMDb")
    parser.add_argument("--limit", type=int, default=100, help="Max items to fetch (dev)")
    args = parser.parse_args()

    api_key = os.getenv("OMDB_API_KEY", "")
    if not api_key:
        print("Set OMDB_API_KEY in .env. Get a key at http://www.omdbapi.com/")
        return 1

    print(f"OMDb key configured. Full fetch for {args.limit} items implemented in Step 5.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
