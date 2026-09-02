from pathlib import Path

import httpx
import pytest
from app.services.poster_download import (
    download_poster,
    existing_poster_item_ids,
    find_existing_poster,
    infer_extension,
    infer_extension_from_url,
    poster_filename,
    run_poster_download,
)


def test_poster_filename():
    assert poster_filename(42) == "42.jpg"
    assert poster_filename(42, "png") == "42.png"


def test_infer_extension_from_url():
    assert infer_extension_from_url("https://image.tmdb.org/t/p/w500/abc.jpg") == "jpg"
    assert infer_extension_from_url("https://example.com/poster.webp") == "webp"
    assert infer_extension_from_url("https://example.com/noext") is None


def test_infer_extension():
    assert infer_extension("image/png", "https://example.com/x") == "png"
    assert infer_extension(None, "https://example.com/poster.jpeg") == "jpg"
    assert infer_extension(None, "https://example.com/noext") == "jpg"


def test_existing_poster_item_ids(tmp_path: Path):
    posters_dir = tmp_path / "posters"
    posters_dir.mkdir()
    (posters_dir / "10.jpg").write_bytes(b"x")
    (posters_dir / "11.png").write_bytes(b"x")
    (posters_dir / "bad.txt").write_bytes(b"x")
    (posters_dir / "empty.jpg").write_bytes(b"")

    assert existing_poster_item_ids(posters_dir) == {10, 11}


def test_find_existing_poster(tmp_path: Path):
    posters_dir = tmp_path / "posters"
    posters_dir.mkdir()
    poster = posters_dir / "10.jpg"
    poster.write_bytes(b"fake")

    assert find_existing_poster(posters_dir, 10) == poster
    assert find_existing_poster(posters_dir, 11) is None


def test_download_poster_writes_file(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"image-bytes", headers={"content-type": "image/jpeg"})

    transport = httpx.MockTransport(handler)
    dest = tmp_path / "1.jpg"

    with httpx.Client(transport=transport) as client:
        assert download_poster("https://example.com/poster.jpg", dest, client) is True

    assert dest.read_bytes() == b"image-bytes"


def test_download_poster_handles_404(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    dest = tmp_path / "1.jpg"

    with httpx.Client(transport=transport) as client:
        assert download_poster("https://example.com/missing.jpg", dest, client) is False

    assert not dest.exists()


@pytest.mark.skipif(
    not Path("backend").exists(),
    reason="integration test needs project layout",
)
def test_run_poster_download_integration(tmp_path: Path, monkeypatch):
    pytest.importorskip("sqlalchemy")
    from app.db.session import SessionLocal
    from app.models.item import Item
    from sqlalchemy import text

    session = SessionLocal()
    try:
        session.execute(text("SELECT 1"))
    except Exception:
        session.close()
        pytest.skip("PostgreSQL not available")

    item_id = 999_001
    try:
        existing = session.get(Item, item_id)
        if existing:
            session.delete(existing)
            session.commit()

        item = Item(
            item_id=item_id,
            title="Poster Test Movie",
            image_url="https://example.com/poster.jpg",
        )
        session.add(item)
        session.commit()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=b"poster-data",
                headers={"content-type": "image/jpeg"},
            )

        posters_dir = tmp_path / "posters"
        original_async_client = httpx.AsyncClient

        class PatchedAsyncClient(original_async_client):
            def __init__(self, *args, **kwargs):
                kwargs["transport"] = httpx.MockTransport(handler)
                super().__init__(*args, **kwargs)

        monkeypatch.setattr("app.services.poster_download.httpx.AsyncClient", PatchedAsyncClient)
        monkeypatch.setattr(
            "app.services.poster_download.items_with_poster_urls",
            lambda session, posters_dir, limit, force=False, existing_ids=None: [
                (item_id, "https://example.com/poster.jpg")
            ],
        )

        counts = run_poster_download(
            session,
            posters_dir=posters_dir,
            limit=1,
            workers=1,
        )

        session.refresh(item)
        assert counts["downloaded"] == 1
        assert item.metadata_json["poster_path"] == str(posters_dir / f"{item_id}.jpg")
        assert (posters_dir / f"{item_id}.jpg").exists()
    finally:
        existing = session.get(Item, item_id)
        if existing:
            session.delete(existing)
            session.commit()
        session.close()
