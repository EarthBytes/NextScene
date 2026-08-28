from app.services.imdb_bulk_enrich import parse_imdb_genres
from app.services.tmdb_metadata import parse_tmdb_movie, TMDB_IMAGE_BASE


def test_parse_imdb_genres():
    assert parse_imdb_genres("Action,Adventure,Comedy") == ["Action", "Adventure", "Comedy"]
    assert parse_imdb_genres("\\N") is None


def test_parse_tmdb_movie():
    data = {
        "overview": "A team struggles to stay together.",
        "poster_path": "/abc123.jpg",
        "tagline": "All roads end here.",
        "release_date": "2017-05-05",
        "vote_average": 7.6,
        "vote_count": 1000,
        "original_language": "en",
        "genres": [{"id": 28, "name": "Action"}],
    }
    parsed = parse_tmdb_movie(data)
    assert parsed is not None
    assert parsed["description"] == "A team struggles to stay together."
    assert parsed["image_url"] == f"{TMDB_IMAGE_BASE}/abc123.jpg"
    assert parsed["genres"] == ["Action"]
    assert parsed["metadata_json"]["tagline"] == "All roads end here."


def test_parse_tmdb_not_found():
    assert parse_tmdb_movie({"success": False}) is None
