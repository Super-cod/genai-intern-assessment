import pytest
from fastapi.testclient import TestClient

import app.api as api

from .conftest import INVOICE_LINES, good_raw, make_pdf


@pytest.fixture
def client(make_service):
    api._service = make_service([good_raw()])
    yield TestClient(api.app)
    api._service = None


def test_openapi_documents_extract_endpoint(client):
    spec = client.get("/openapi.json").json()
    assert "/extract" in spec["paths"]
    assert {"413", "415", "422", "503"} <= set(spec["paths"]["/extract"]["post"]["responses"])


def test_extract_ok(client):
    r = client.post("/extract", files={"file": ("a.pdf", make_pdf(INVOICE_LINES), "application/pdf")})
    assert r.status_code == 200 and r.json()["status"] == "complete"


@pytest.mark.parametrize("name,data,code,err", [
    ("a.txt", b"plain text", 415, "unsupported_format"),
    ("a.pdf", b"", 400, "empty_file"),
    ("a.pdf", b"%PDF-1.4\n" + b"0" * (2 * 1024 * 1024), 413, "file_too_large"),
])
def test_extract_errors(client, name, data, code, err):
    r = client.post("/extract", files={"file": (name, data, "application/octet-stream")})
    assert r.status_code == code and r.json()["error"] == err


def test_missing_file_is_422(client):
    assert client.post("/extract").status_code == 422


def test_document_types(client):
    assert "invoice" in client.get("/document-types").json()
