from app.services.omdb_metadata import parse_omdb_response

SAMPLE_RESPONSE = {
    "Title": "Guardians of the Galaxy: Vol. 2",
    "Year": "2017",
    "Rated": "PG-13",
    "Released": "05 May 2017",
    "Runtime": "136 min",
    "Genre": "Action, Adventure, Comedy",
    "Director": "James Gunn",
    "Writer": "James Gunn, Dan Abnett, Andy Lanning",
    "Actors": "Chris Pratt, Zoe Saldaña, Dave Bautista",
    "Plot": "The Guardians struggle to keep together as a team while dealing with their personal family issues.",
    "Language": "English",
    "Country": "United States",
    "Awards": "Nominated for 1 Oscar. 15 wins & 62 nominations total",
    "Poster": "https://m.media-amazon.com/images/M/MV5BNWE5MGI3MDctMmU5Ni00YzI2LWEzMTQtZGIyZDA5MzQzNDBhXkEyXkFqcGc@._V1_QL75_UX380_CR0,1,380,562_.jpg",
    "Ratings": [
        {"Source": "Internet Movie Database", "Value": "7.6/10"},
        {"Source": "Rotten Tomatoes", "Value": "85%"},
    ],
    "Metascore": "67",
    "imdbRating": "7.6",
    "imdbVotes": "828,114",
    "imdbID": "tt3896198",
    "Type": "movie",
    "BoxOffice": "$389,813,101",
    "Response": "True",
}


def test_parse_omdb_response():
    parsed = parse_omdb_response(SAMPLE_RESPONSE)
    assert parsed is not None
    assert parsed["description"].startswith("The Guardians struggle")
    assert parsed["image_url"].startswith("https://m.media-amazon.com")
    assert parsed["genres"] == ["Action", "Adventure", "Comedy"]
    assert parsed["metadata_json"]["director"] == "James Gunn"
    assert parsed["metadata_json"]["imdb_rating"] == "7.6"


def test_parse_omdb_not_found():
    assert parse_omdb_response({"Response": "False"}) is None
