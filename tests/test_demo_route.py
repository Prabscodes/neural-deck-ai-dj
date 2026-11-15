from fastapi.testclient import TestClient

from autodj_server import create_app


def test_demo_redirects_legacy_url_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("AUTODJ_ENABLE_LEGACY_STUDIO", raising=False)
    client = TestClient(create_app(cache_dir=str(tmp_path)))
    response = client.get("/studio", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/"
    assert "Start listening" in client.get(response.headers["location"]).text
