from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
import logging
from pathlib import Path
import secrets
import time

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .service import InputError, MAX_IMAGE_BYTES, ModelService, SessionExpired

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger(__name__)


class EstimateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    session_id: str = Field(min_length=20, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    model: str = Field(min_length=1, max_length=150)
    year: int = Field(strict=True, ge=1990, le=2100)
    mileage_km: float = Field(ge=0, le=1_000_000)
    color: str = Field(min_length=1, max_length=80)
    body: str = Field(min_length=1, max_length=80)


def create_app(service=None):
    @asynccontextmanager
    async def lifespan(application):
        async def clean_sessions():
            while True:
                await asyncio.sleep(60)
                await asyncio.to_thread(application.state.service.cleanup)

        loader = None
        if application.state.service.status == "loading":
            loader = asyncio.create_task(asyncio.to_thread(application.state.service.load))
        cleaner = asyncio.create_task(clean_sessions())
        yield
        cleaner.cancel()
        with suppress(asyncio.CancelledError):
            await cleaner
        if loader:
            await loader
        application.state.service.sessions.clear()

    application = FastAPI(title="AVTO API", description="Оценка автомобиля по фото и данным DVM-CAR", version="1.0.0", lifespan=lifespan, docs_url=None, redoc_url=None)
    application.state.service = service or ModelService(ROOT)
    application.state.worker = asyncio.Semaphore(1)

    @application.middleware("http")
    async def response_headers(request, call_next):
        length = request.headers.get("content-length")
        if length and (not length.isdigit() or int(length) > MAX_IMAGE_BYTES + 1024 * 1024):
            return JSONResponse({"detail": "Файл слишком большой. Максимум — 15 МБ."}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if "Content-Security-Policy" not in response.headers:
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data: blob:; font-src 'self'; connect-src 'self'; "
                "frame-ancestors 'self'; base-uri 'self'; form-action 'self'"
            )
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @application.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        return JSONResponse({"detail": "Проверь данные: год — целое число от 1990 до текущего года, пробег — от 0 до 1 000 000 км. Выбери модель, цвет и кузов из списка."}, status_code=422)

    def ready():
        instance = application.state.service
        if instance.status != "ready":
            detail = "Модели ещё загружаются. Подожди немного и попробуй снова." if instance.status == "loading" else "Не удалось запустить модели. Перезапусти приложение из папки проекта."
            raise HTTPException(status_code=503, detail=detail)
        return instance

    @application.get("/api/health")
    async def health():
        status = application.state.service.status
        return {"status": status, "message": {"ready": "Готов к оценке", "loading": "Подготавливаем модели", "error": "Не удалось загрузить модели"}[status]}

    @application.get("/api/metadata")
    async def metadata():
        return await asyncio.to_thread(ready().metadata)

    @application.post("/api/recognize")
    async def recognize(photo: UploadFile = File(...)):
        instance = ready()
        try:
            content = await photo.read(MAX_IMAGE_BYTES + 1)
        finally:
            await photo.close()
        if len(content) > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Фотография должна быть не больше 15 МБ.")
        start = time.monotonic()
        try:
            async with application.state.worker:
                result = await asyncio.to_thread(instance.recognize, content)
            result["processing_seconds"] = round(time.monotonic() - start, 2)
            return result
        except InputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("Recognition failed")
            raise HTTPException(status_code=500, detail="Не удалось распознать автомобиль. Попробуй другое фото спереди или в три четверти.") from exc

    @application.post("/api/estimate")
    async def estimate(data: EstimateRequest):
        instance = ready()
        try:
            async with application.state.worker:
                return await asyncio.to_thread(instance.estimate, data.session_id, data.model, data.year,
                                               data.mileage_km, data.color, data.body)
        except SessionExpired as exc:
            raise HTTPException(status_code=410, detail=str(exc)) from exc
        except InputError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("Valuation failed")
            raise HTTPException(status_code=500, detail="Не удалось рассчитать стоимость. Проверь выбранные характеристики и попробуй ещё раз.") from exc

    @application.delete("/api/sessions/{session_id}", status_code=204)
    async def delete_session(session_id: str):
        application.state.service.delete_session(session_id)

    @application.get("/", include_in_schema=False)
    async def index():
        return FileResponse(ROOT / "frontend" / "index.html", headers={"Cache-Control": "no-cache"})

    @application.get("/docs", include_in_schema=False)
    async def docs():
        nonce = secrets.token_urlsafe(18)
        response = get_swagger_ui_html(
            openapi_url="/openapi.json", title="AVTO — API",
            swagger_js_url="/static/vendor/swagger-ui-bundle.js",
            swagger_css_url="/static/vendor/swagger-ui.css",
            swagger_favicon_url="/static/assets/favicon.svg",
        )
        html = response.body.decode("utf-8").replace("<script>", f'<script nonce="{nonce}">')
        policy = (
            f"default-src 'self'; script-src 'self' 'nonce-{nonce}'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'self'; base-uri 'self'"
        )
        return HTMLResponse(html, headers={"Content-Security-Policy": policy, "Cache-Control": "no-store"})

    application.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")
    return application


app = create_app()
