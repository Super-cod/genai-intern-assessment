import pytest
from fastapi.testclient import TestClient

import app.api as api

from .conftest import MANUAL


@pytest.fixture
def client(make_pipeline):
    api._pipeline = make_pipeline([{"answerable": True, "answer": "It lasts 8 hours [S1].", "cited_sources": ["S1"]}])
    yield TestClient(api.app)
    api._pipeline = None


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_upload_mixed_batch_reports_per_file(client):
    files = [("files", ("manual.txt", MANUAL.encode(), "text/plain")),
             ("files", ("fake.pdf", b"not a pdf", "application/pdf")),
             ("files", ("data.csv", b"a,b", "text/csv"))]
    r = client.post("/documents", files=files)
    assert r.status_code == 200
    res = {x["filename"]: x for x in r.json()["results"]}
    assert res["manual.txt"]["ok"]
    assert res["fake.pdf"]["error_code"] == "corrupt_document"
    assert res["data.csv"]["error_code"] == "unsupported_file_type"


def test_oversized_upload_rejected(client, monkeypatch):
    monkeypatch.setattr(api.get_settings(), "max_upload_mb", 0)
    r = client.post("/documents", files=[("files", ("manual.txt", MANUAL.encode(), "text/plain"))])
    assert r.json()["results"][0]["error_code"] == "file_too_large"
    monkeypatch.undo()


def test_query_without_documents_is_409(client):
    r = client.post("/query", json={"question": "hello?"})
    assert r.status_code == 409 and r.json()["error"] == "no_documents_indexed"


def test_empty_question_is_422(client):
    assert client.post("/query", json={"question": ""}).status_code == 422


def test_query_returns_structured_answer(client):
    client.post("/documents", files=[("files", ("manual.txt", MANUAL.encode(), "text/plain"))])
    r = client.post("/query", json={"question": "How long does the battery last?"})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "answered"
    assert body["citations"][0]["doc_name"] == "manual.txt" and "timings_ms" in body


def test_delete_unknown_document_is_404(client):
    assert client.delete("/documents/nope").status_code == 404
