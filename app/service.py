"""Cached web adapter for the repository's train.py and predict.py models."""
from __future__ import annotations

import base64
import datetime as dt
import importlib
import io
import logging
import math
import os
from pathlib import Path
import secrets
import sys
import threading
import time

LOGGER = logging.getLogger(__name__)
SESSION_TTL = 30 * 60
MAX_SESSIONS = 32
MAX_IMAGE_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 30_000_000
KM_PER_MILE = 1.609344

COLORS_RU = {
    "Beige": "Бежевый", "Black": "Чёрный", "Blue": "Синий", "Bronze": "Бронзовый",
    "Brown": "Коричневый", "Burgundy": "Бордовый", "Gold": "Золотистый", "Green": "Зелёный",
    "Grey": "Серый", "Indigo": "Индиго", "Magenta": "Малиновый", "Maroon": "Тёмно-бордовый",
    "Multicolour": "Многоцветный", "Orange": "Оранжевый", "Pink": "Розовый", "Purple": "Фиолетовый",
    "Red": "Красный", "Silver": "Серебристый", "Turquoise": "Бирюзовый", "Unlisted": "Другой",
    "White": "Белый", "Yellow": "Жёлтый", "Navy": "Тёмно-синий",
}
BODY_RU = {
    "Car Derived Van": "Легковой фургон", "Convertible": "Кабриолет", "Coupe": "Купе",
    "Estate": "Универсал", "Hatchback": "Хэтчбек", "Limousine": "Лимузин", "MPV": "Минивэн",
    "Panel Van": "Фургон", "SUV": "Внедорожник / кроссовер", "Saloon": "Седан",
    "Pickup": "Пикап", "Combi Van": "Грузопассажирский фургон", "Camper": "Автодом",
    "Minibus": "Микроавтобус", "Window Van": "Фургон с остеклением",
}


class InputError(ValueError):
    """A user-facing input error."""


class SessionExpired(InputError):
    pass


def native(value):
    """Return finite JSON-native values, including nested numpy scalars."""
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if hasattr(value, "item"):
        return native(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class ModelService:
    def __init__(self, root: Path):
        self.root = root
        self.status = "loading"
        self.load_error = None
        self.sessions = {}
        self.session_lock = threading.RLock()
        self.work_lock = threading.Lock()
        self.rate_lock = threading.Lock()
        self.rate = None
        self.rate_checked_at = 0.0
        self.background_session = None

    def load(self):
        try:
            cache = self.root / ".cache"
            os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
            os.environ.setdefault("U2NET_HOME", str(self.root / "models" / "rembg"))
            os.environ.setdefault("NUMBA_CACHE_DIR", str(cache / "numba"))
            os.environ.setdefault("OMP_NUM_THREADS", "4")
            source = str(self.root)
            if source not in sys.path:
                sys.path.insert(0, source)
            import numpy as np
            import pandas as pd
            import torch
            from PIL import Image, ImageOps, UnidentifiedImageError
            from torchvision import transforms as transforms

            self.np, self.pd, self.torch = np, pd, torch
            self.Image, self.ImageOps = Image, ImageOps
            self.UnidentifiedImageError = UnidentifiedImageError
            self.original = importlib.import_module("train")
            # The original script saved this custom class as __main__.PriceModel.
            # Restore that name for loading; the serialized model is not altered.
            setattr(sys.modules["__main__"], "PriceModel", self.original.PriceModel)
            setattr(sys.modules["__main__"], "Specs", self.original.Specs)
            self.predict_module = importlib.import_module("predict")
            self.predict_module.HERE = str(self.root)
            torch.set_num_threads(min(4, os.cpu_count() or 1))
            checkpoint = torch.load(self.root / "car_model.pt", map_location="cpu", weights_only=False)
            self.load_recognizer(checkpoint)
            size = checkpoint["img_size"]
            self.transform = transforms.Compose([
                transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            self.pricer = self.predict_module.Pricer()
            self.models = sorted(str(v) for v in self.pricer.market.model.unique())
            self.colors = sorted(set(self.classes["color"]) | {str(v) for v in self.pricer.market.color.dropna().unique()})
            self.bodies = sorted(set(self.classes["body"]) | {str(v) for v in self.pricer.market.body.dropna().unique()})
            from rembg import new_session
            self.background_session = new_session("u2net", providers=["CPUExecutionProvider"])
            self.refresh_rate()
            self.status = "ready"
            LOGGER.info("Models loaded: %d image classes, %d price models", len(self.classes["model"]), len(self.models))
        except (Exception, SystemExit) as exc:
            self.load_error = str(exc)
            self.status = "error"
            LOGGER.exception("Model startup failed")

    def load_recognizer(self, checkpoint):
        """Accept the legacy single-network checkpoint and the current ensemble."""
        self.classes = {key: [str(v) for v in checkpoint["classes"][key]] for key in ("model", "color", "body")}
        members = checkpoint.get("members")
        if not members:
            if "weights" not in checkpoint:
                raise InputError("В car_model.pt нет весов распознавания.")
            members = [{"arch": "convnext_tiny", "weights": checkpoint["weights"]}]
        self.nets = []
        for member in members:
            net = self.original.CarNet(*(len(self.classes[k]) for k in ("model", "color", "body")),
                                       pretrained=False, arch=member["arch"])
            net.load_state_dict(member["weights"])
            net.eval()
            self.nets.append(net)

    def predict_probabilities(self, x):
        """Average probabilities, matching the current predict.recognize algorithm."""
        with self.torch.inference_mode():
            predictions = [[self.torch.softmax(o, 1)[0] for o in net(x)] for net in self.nets]
            return [self.torch.stack([p[i] for p in predictions]).mean(0).numpy() for i in range(3)]

    def refresh_rate(self):
        with self.rate_lock:
            if self.rate is not None and time.monotonic() - self.rate_checked_at < 6 * 3600:
                return dict(self.rate)
            rate, date, source = self.predict_module.gbp_rate()
            self.rate = {"gbp_rub": rate, "date": date, "source": source,
                         "is_fallback": source != "ЦБ РФ"}
            self.rate_checked_at = time.monotonic()
            return dict(self.rate)

    def metadata(self):
        return native({
            "models": self.models,
            "colors": [{"value": v, "label": COLORS_RU.get(v, v)} for v in self.colors],
            "bodies": [{"value": v, "label": BODY_RU.get(v, v)} for v in self.bodies],
            "quality": self.pricer.quality,
            "recognition_models": len(self.classes["model"]),
            "pricing_models": len(self.models),
            "current_year": dt.date.today().year,
            "rate": self.refresh_rate(),
            "max_file_mb": MAX_IMAGE_BYTES // 1024 // 1024,
        })

    def cleanup(self):
        with self.session_lock:
            expired = [key for key, value in self.sessions.items() if time.monotonic() - value["created_at"] >= SESSION_TTL]
            for key in expired:
                del self.sessions[key]

    def delete_session(self, token):
        with self.session_lock:
            self.sessions.pop(token, None)

    def session(self, token):
        self.cleanup()
        with self.session_lock:
            entry = self.sessions.get(token)
            if entry is None:
                raise SessionExpired("Сессия завершилась. Загрузи фотографию ещё раз.")
            return dict(entry)

    def read_image(self, content: bytes):
        if not content:
            raise InputError("Выбери фотографию автомобиля.")
        if len(content) > MAX_IMAGE_BYTES:
            raise InputError("Фотография должна быть не больше 15 МБ.")
        try:
            with self.Image.open(io.BytesIO(content)) as check:
                if check.format not in ("JPEG", "PNG", "WEBP"):
                    raise InputError("Поддерживаются фотографии JPG, PNG и WebP.")
                width, height = check.size
                if min(width, height) < 64:
                    raise InputError("Фотография слишком маленькая. Нужен размер хотя бы 64 × 64 пикселя.")
                if width * height > MAX_IMAGE_PIXELS:
                    raise InputError("Слишком большое разрешение. Уменьши фото до 30 мегапикселей.")
                check.verify()
            with self.Image.open(io.BytesIO(content)) as opened:
                image = self.ImageOps.exif_transpose(opened).convert("RGB")
                image.load()
                return image
        except InputError:
            raise
        except (self.UnidentifiedImageError, OSError, ValueError, self.Image.DecompressionBombError) as exc:
            raise InputError("Не удалось прочитать фотографию. Попробуй другой JPG, PNG или WebP.") from exc

    def prepare_photo(self, image):
        # Same u2net removal, alpha threshold, square padding and resize as predict.py.
        from rembg import remove
        background_removed = True
        try:
            cut = remove(image.convert("RGB"), session=self.background_session)
            box = cut.split()[-1].point(lambda alpha: 255 if alpha > 20 else 0).getbbox()
            image = self.original.to_rgb(cut.crop(box)) if box else image.convert("RGB")
        except Exception:
            LOGGER.exception("Background removal failed; using original fallback")
            image = image.convert("RGB")
            background_removed = False
        side = max(image.size)
        square = self.Image.new("RGB", (side, side), (255, 255, 255))
        square.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        return square.resize((300, 300)), background_removed

    def recognize(self, content: bytes):
        image = self.read_image(content)
        with self.work_lock:
            prepared, background_removed = self.prepare_photo(image)
            x = self.transform(prepared)[None]
            outputs = self.predict_probabilities(x)
            recognition = {
                key: [{"value": self.classes[key][int(i)], "label": self.label(key, self.classes[key][int(i)]),
                       "confidence": float(probabilities[i])} for i in self.np.argsort(-probabilities)[:3]]
                for key, probabilities in zip(("model", "color", "body"), outputs)
            }
        preview = io.BytesIO()
        prepared.save(preview, format="JPEG", quality=90)
        token = secrets.token_urlsafe(24)
        self.cleanup()
        with self.session_lock:
            if len(self.sessions) >= MAX_SESSIONS:
                oldest = min(self.sessions, key=lambda key: self.sessions[key]["created_at"])
                del self.sessions[oldest]
            self.sessions[token] = {"created_at": time.monotonic(), "recognition": recognition,
                                    "background_removed": background_removed}
        return native({"session_id": token, "recognition": recognition,
                       "prepared_image": "data:image/jpeg;base64," + base64.b64encode(preview.getvalue()).decode("ascii"),
                       "background_removed": background_removed, "expires_in_seconds": SESSION_TTL})

    @staticmethod
    def label(key, value):
        return COLORS_RU.get(value, value) if key == "color" else BODY_RU.get(value, value) if key == "body" else value

    def estimate(self, token, model, year, km, color, body):
        entry = self.session(token)
        if model not in self.models:
            raise InputError("Выбери автомобиль из списка доступных моделей.")
        if color not in self.colors or body not in self.bodies:
            raise InputError("Проверь выбранные цвет и тип кузова.")
        today = dt.date.today()
        if isinstance(year, bool) or not 1990 <= year <= today.year:
            raise InputError("Проверь год выпуска автомобиля.")
        if not math.isfinite(km) or not 0 <= km <= 1_000_000:
            raise InputError("Пробег должен быть от 0 до 1 000 000 км.")
        with self.work_lock:
            result = self._analytics(entry, model, year, km, color, body, today)
        return native(result)

    def _analytics(self, entry, model, year, km, color, body, today):
        np, pd, pricer = self.np, self.pd, self.pricer
        miles = km / KM_PER_MILE
        p, lo, hi = [float(v[0]) for v in pricer.price(model, year, miles, color, body, now=today)]
        if not all(math.isfinite(v) and v > 0 for v in (p, lo, hi)):
            raise RuntimeError("The model returned an invalid price")
        mk, gid, maker = pricer.info(model)
        rate = self.refresh_rate()
        fx = rate["gbp_rub"]
        typical_miles = float((mk.mileage / mk.age).median())
        own_miles = miles / max(today.year - year + .5, .5)
        years = np.arange(max(year - 6, int(mk.reg_year.min())), min(today.year, max(int(mk.reg_year.max()), year)) + 1)
        curve = pricer.price(model, years, typical_miles * (today.year - years + .5), color, body, now=today)[0]
        future_gbp = [p] + [float(pricer.price(model, year, miles + own_miles * offset, color, body,
                             now=dt.date(today.year + offset, today.month, 1))[0][0]) for offset in (1, 2, 3)]
        # Compare the current price with the SAME car after another 10,000 km.
        # No change is made to the price model or its feature engineering.
        with_10k = float(pricer.price(model, year, miles + 10_000 / KM_PER_MILE, color, body, now=today)[0][0])
        palette = mk.color.value_counts(normalize=True).head(7)
        colors = [{"value": str(c), "label": COLORS_RU.get(str(c), str(c)), "share": float(share),
                   "effect_percent": (float(pricer.price(model, year, miles, c, body, now=today)[0][0]) / p - 1) * 100,
                   "selected": c == color} for c, share in palette.items()]
        comparable = mk[(mk.age.between(today.year - year - 1, today.year - year + 2)) &
                        (mk.mileage.between(miles * .7, miles * 1.3 + 3000))]
        new_table = pricer.table.get(str(gid), {})
        new_gbp = self.original.new_price(pricer.table, gid, year)
        sales = pd.DataFrame()
        if pricer.sales is not None:
            sales = pricer.sales[pricer.sales.gid == str(gid)].sort_values("year")
        warnings = []
        if hasattr(pricer, "specs"):
            warnings.append("Объём двигателя, коробка и топливо не указаны в веб-форме. "
                            "Использованы типичные значения для выбранной модели из обучающих данных.")
        confidence = entry["recognition"]["model"][0]["confidence"]
        if model == entry["recognition"]["model"][0]["value"] and confidence < .4:
            warnings.append("Распознавание неуверенное. Проверь модель автомобиля перед использованием оценки.")
        if year > 2021:
            warnings.append("Год выпуска новее обучающих данных (до 2021 года). Погрешность может быть выше.")
        if year < 2008 or today.year - year > 14:
            warnings.append("Возраст автомобиля выходит за основной диапазон обучающих данных. Оценка менее надёжна.")
        if km > 250_000 * KM_PER_MILE:
            warnings.append("Пробег выше диапазона обучающих данных. Оценка менее надёжна.")
        if not entry["background_removed"]:
            warnings.append("Фон не удалось удалить автоматически. Распознавание выполнено по исходной фотографии.")
        if rate["is_fallback"]:
            warnings.append("Курс ЦБ временно недоступен: использован запасной курс от " + rate["date"] + ".")
        comparison = None
        if len(comparable) >= 5:
            comparison = {"count": len(comparable),
                          "q25_rub": float(comparable.price_today.quantile(.25)) * fx,
                          "median_rub": float(comparable.price_today.median()) * fx,
                          "q75_rub": float(comparable.price_today.quantile(.75)) * fx}
        return {
            "vehicle": {"model": model, "maker": str(maker), "year": year, "mileage_km": km,
                        "color": color, "color_label": COLORS_RU.get(color, color),
                        "body": body, "body_label": BODY_RU.get(body, body),
                        "manually_corrected": any([model != entry["recognition"]["model"][0]["value"],
                                                   color != entry["recognition"]["color"][0]["value"],
                                                   body != entry["recognition"]["body"][0]["value"]])},
            "valuation": {"gbp": p, "rub": p * fx, "low_rub": lo * fx, "high_rub": hi * fx,
                          "interval_label": "Диапазон P10–P90", "date": today.isoformat(), "rate": rate},
            "recognition": entry["recognition"], "warnings": warnings,
            "insights": {"typical_km_year": typical_miles * KM_PER_MILE, "own_km_year": own_miles * KM_PER_MILE,
                         "change_per_10k_rub": (with_10k - p) * fx,
                         "one_year_change_percent": (future_gbp[1] / p - 1) * 100,
                         "new_price_gbp": float(new_gbp) if math.isfinite(new_gbp) else None,
                         "retained_percent": p / new_gbp * 100 if math.isfinite(new_gbp) and new_gbp > 0 else None},
            "charts": {
                "by_year": [{"year": int(y), "rub": float(value) * fx} for y, value in zip(years, curve)],
                "selected_year": {"year": year, "rub": p * fx},
                "forecast": [{"offset": i, "year": today.year + i, "rub": value * fx} for i, value in enumerate(future_gbp)],
                "colors": colors,
                "new_prices": [{"year": int(y), "gbp": float(value)} for y, value in sorted(new_table.items())],
                "sales": [{"year": int(r.year), "count": float(r.sales)} for r in sales.itertuples()
                          if pd.notna(r.sales) and math.isfinite(float(r.sales))],
            },
            "comparables": comparison, "quality": pricer.quality,
            "method": {"market": "Великобритания", "currency": "GBP → RUB", "dataset": "DVM-CAR",
                       "data_years": "2018–2021", "price_base_year": 2026,
                       "recognition_models": len(self.classes["model"]), "pricing_models": len(self.models)},
        }
