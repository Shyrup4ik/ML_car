"""Exercise upload validation, request validation and session isolation without ML."""
import datetime as dt
import io
from pathlib import Path
import secrets
import threading

from fastapi.testclient import TestClient
from PIL import Image, ImageOps, UnidentifiedImageError
import pytest

from app.main import create_app
from app.service import MAX_IMAGE_BYTES, ModelService


@pytest.fixture
def service():
    instance = ModelService(Path(__file__).resolve().parents[1])
    instance.status = "ready"
    instance.Image, instance.ImageOps = Image, ImageOps
    instance.UnidentifiedImageError = UnidentifiedImageError
    instance.models = ["Audi A1", "Audi A3"]
    instance.colors = ["Black", "Silver"]
    instance.bodies = ["Hatchback", "Saloon"]
    instance._analytics = lambda entry, model, year, km, color, body, today: {
        "vehicle": {"model": model, "year": year, "mileage_km": km, "color": color, "body": body}
    }

    def recognize(content):
        instance.read_image(content)
        token = secrets.token_urlsafe(24)
        instance.sessions[token] = {"created_at": __import__("time").monotonic()}
        return {"session_id": token, "recognition": {"model": [{"value": "Audi A1", "confidence": .9}]}}

    instance.recognize = recognize
    instance.metadata = lambda: {"models": instance.models}
    return instance


@pytest.fixture
def client(service):
    with TestClient(create_app(service)) as instance:
        yield instance


def photograph():
    output = io.BytesIO()
    Image.new("RGB", (100, 100), "silver").save(output, "JPEG")
    return output.getvalue()


def recognize(client):
    response = client.post("/api/recognize", files={"photo": ("car.jpg", photograph(), "image/jpeg")})
    assert response.status_code == 200
    return response.json()["session_id"]


def valuation(token, **changes):
    data = {"session_id": token, "model": "Audi A1", "year": 2017, "mileage_km": 85000,
            "color": "Black", "body": "Hatchback"}
    return data | changes


def test_actual_page_and_local_assets_are_served(client):
    for path in ["/", "/static/app.js", "/static/styles.css", "/static/assets/car.svg"]:
        response = client.get(path)
        assert response.status_code == 200
        assert len(response.content) > 100
    assert '<html lang="ru">' in client.get("/").text
    assert "default-src 'self'" in client.get("/").headers["content-security-policy"]


def test_api_documentation_uses_local_assets_and_script_nonce(client):
    response = client.get("/docs")
    assert response.status_code == 200
    assert '/static/vendor/swagger-ui-bundle.js' in response.text
    assert '<script nonce="' in response.text
    assert "'nonce-" in response.headers["content-security-policy"]
    assert 'src="https://' not in response.text


def test_loading_is_reported_without_running_inference(client, service):
    service.status = "loading"
    assert client.get("/api/health").json()["status"] == "loading"
    assert client.get("/api/metadata").status_code == 503
    assert client.post("/api/recognize", files={"photo": ("car.jpg", photograph())}).status_code == 503


@pytest.mark.parametrize("content", [b"", b"not a photograph", b"<svg xmlns='http://www.w3.org/2000/svg'></svg>"])
def test_invalid_contents_cannot_bypass_checks_with_image_filename(client, content):
    response = client.post("/api/recognize", files={"photo": ("car.jpg", content, "image/jpeg")})
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], str)


def test_tiny_image_rejected(client):
    output = io.BytesIO()
    Image.new("RGB", (20, 20)).save(output, "PNG")
    assert client.post("/api/recognize", files={"photo": ("tiny.png", output.getvalue())}).status_code == 422


def test_oversized_upload_rejected(client):
    response = client.post("/api/recognize", files={"photo": ("large.jpg", b"x" * (MAX_IMAGE_BYTES + 1))})
    assert response.status_code == 413


@pytest.mark.parametrize("changes", [
    {"year": "2017"}, {"year": True}, {"year": 1989}, {"year": dt.date.today().year + 1},
    {"mileage_km": -1}, {"mileage_km": 1000001}, {"model": "Unknown"},
    {"color": "Unknown"}, {"body": "Unknown"}, {"unexpected": "field"},
])
def test_invalid_estimate_never_reaches_model(client, changes):
    token = recognize(client)
    response = client.post("/api/estimate", json=valuation(token, **changes))
    assert response.status_code == 422


def test_manual_correction_and_deletion(client):
    token = recognize(client)
    response = client.post("/api/estimate", json=valuation(token, model="Audi A3", color="Silver", body="Saloon"))
    assert response.status_code == 200
    assert response.json()["vehicle"]["model"] == "Audi A3"
    assert response.json()["vehicle"]["color"] == "Silver"
    assert client.delete("/api/sessions/" + token).status_code == 204
    assert client.post("/api/estimate", json=valuation(token)).status_code == 410


def test_two_sessions_are_independent(client):
    first, second = recognize(client), recognize(client)
    assert first != second
    client.delete("/api/sessions/" + first)
    assert client.post("/api/estimate", json=valuation(second)).status_code == 200
    assert client.post("/api/estimate", json=valuation(first)).status_code == 410


def test_expired_session_is_cleared(client, service):
    token = recognize(client)
    service.sessions[token]["created_at"] -= 1801
    assert client.post("/api/estimate", json=valuation(token)).status_code == 410
    assert token not in service.sessions


def test_nonfinite_numbers_rejected(client):
    token = recognize(client)
    payload = __import__("json").dumps(valuation(token)).replace('85000', 'NaN')
    response = client.post("/api/estimate", content=payload, headers={"content-type": "application/json"})
    assert response.status_code == 422
