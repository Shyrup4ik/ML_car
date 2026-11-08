"""
ОБУЧЕНИЕ: оценка цены автомобиля по фото (датасет DVM-CAR)

Датасет DVM-CAR кладётся в папку data/ рядом с этим файлом (подробно — data/PUT_DATASET_HERE.txt):
    data/Ad_table.csv, Ad_table_extra.csv, Price_table.csv, Sales_table.csv, Image_table.csv
    data/Confirmed_fronts.zip (или resized_DVM_v2.zip) — архив с фото, распаковывать не нужно

Запуск:  python train.py
         python train.py --price-only   (только модель цены: без фото и видеокарты)
Результат (в папке проекта): car_model.pt, price_model.joblib, metrics.txt, training_curves.png

Что делает:
  1. Отбирает до 50 000 фото машин 2013–2020 годов (вид спереди, 1 фото на объявление).
  2. Делит данные на train / val / test (70/15/15) по объявлениям — фото одной машины не попадают в разные части.
  3. Модель цены: CatBoost, гиперпараметры подбираются Optuna по валидации.
  4. Нейросеть: ансамбль из трёх сетей (ConvNeXt-Tiny, EfficientNet-V2-S, ResNet-50, предобучены на ImageNet),
     у каждой три выхода — модель, цвет, кузов. Гиперпараметры подбираются Optuna отдельно для каждой сети,
     обучение с ранней остановкой; ответ ансамбля — среднее вероятностей трёх сетей.
  5. Считает точность на тесте (тест не участвует ни в подборе, ни в обучении).
"""
import glob
import hashlib
import os
import shutil
import sys
import time
import zipfile

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from catboost import CatBoostRegressor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision import models as tvm

# =============================== НАСТРОЙКИ ===============================
ROOT = os.path.dirname(os.path.abspath(__file__))       # папка проекта: сюда сохраняются модели и отчёты
DATA_DIR = os.path.join(ROOT, "data")                    # сюда кладётся датасет DVM-CAR (CSV и архивы с фото)
YEAR_MIN, YEAR_MAX = 2013, 2020            # годы выпуска машин на фото
PRICE_YEAR_MIN, PRICE_YEAR_MAX = 2008, 2021  # годы для модели цены (шире: нужны и старые машины)
MAX_IMAGES = 50_000                        # максимум фото
MIN_PER_MODEL, MAX_PER_MODEL = 150, 600    # фото на одну модель авто
VIEWPOINT = 0                              # 0 = вид спереди (колонка Predicted_viewpoint)

TUNE = True                                # подбирать гиперпараметры (Optuna)
PRICE_TRIALS = 25                          # попыток для модели цены (CatBoost)
CNN_TRIALS = 8                             # попыток на КАЖДУЮ сеть ансамбля (каждая — 3 эпохи на 30 % фото)

# Гиперпараметры по умолчанию (стартовая точка подбора)
PRICE_PARAMS = dict(learning_rate=0.08, depth=8, l2_leaf_reg=3.0, random_strength=1.0, bagging_temperature=1.0)
PRICE_ITERS = 5000                         # максимум деревьев CatBoost (дальше — ранняя остановка)
CNN_PARAMS = dict(lr=1e-4, lr_head=1e-3, weight_decay=0.05, label_smoothing=0.1, dropout=0.2)
# Сети ансамбля: архитектура -> размер батча (для RTX 4070, 12 ГБ). Убрать сеть — удалить строку.
ENSEMBLE = {"convnext_tiny": 96, "efficientnet_v2_s": 64, "resnet50": 96}
EPOCHS, PATIENCE, BATCH, IMG_SIZE = 15, 3, 96, 224
NUM_WORKERS = 4                            # если зависает на Windows — поставьте 0

# Индекс цен на подержанные авто в Великобритании (ONS, ряд D7E9, 2015=100) — перевод цен в сегодняшние
UK_INDEX = {2015: 100.0, 2016: 95.0, 2017: 93.6, 2018: 95.5, 2019: 93.3, 2020: 96.6, 2021: 108.8,
            2022: 121.5, 2023: 118.9, 2024: 111.2, 2025: 112.0, 2026: 109.9}
SEED = 42
QUICK = "--quick" in sys.argv              # быстрый режим для проверки кода
if QUICK:
    MIN_PER_MODEL, PRICE_TRIALS, CNN_TRIALS, EPOCHS, IMG_SIZE, BATCH, NUM_WORKERS = 30, 2, 2, 2, 64, 16, 0
    PRICE_ITERS, ENSEMBLE = 150, {k: 16 for k in ENSEMBLE}
PRETRAINED = "--no-pretrained" not in sys.argv
# ========================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
try:                                       # CatBoost видит видеокарту сам (нужна NVIDIA + драйвер с CUDA)
    from catboost.utils import get_gpu_device_count
    PRICE_GPU = get_gpu_device_count() > 0 and "--cpu" not in sys.argv
except Exception:
    PRICE_GPU = False
optuna.logging.set_verbosity(optuna.logging.WARNING)
torch.manual_seed(SEED)
np.random.seed(SEED)


def path(name):
    """Файл результата в папке проекта (модели, метрики, best_params.json)."""
    return os.path.join(ROOT, name)


def data_files(pattern):
    """Файлы датасета: в data/ (с подпапками), а для совместимости — и прямо в папке проекта."""
    found = glob.glob(os.path.join(DATA_DIR, "**", pattern), recursive=True) + glob.glob(os.path.join(ROOT, pattern))
    return sorted(set(found))


# ------------------------------------------------------------------ 1. ДАННЫЕ
_UNPACKED = False


def unpack_tables():
    """Достаёт CSV-таблицы из ВСЕХ zip-архивов в data/ (любого размера, из любых подпапок внутри архива,
    включая вложенные zip) прямо в data/. Фото не распаковываются. Делается один раз."""
    global _UNPACKED
    if _UNPACKED:
        return
    _UNPACKED = True
    os.makedirs(DATA_DIR, exist_ok=True)
    queue = data_files("*.zip")
    while queue:
        z = queue.pop(0)
        try:
            zf = zipfile.ZipFile(z)
        except (zipfile.BadZipFile, OSError):
            print(f"  [!] архив {os.path.basename(z)} повреждён или не докачан — скачайте заново")
            continue
        got = []
        for m in zf.namelist():
            base = os.path.basename(m)
            if "__MACOSX" in m or base.startswith("._") or not base.lower().endswith((".csv", ".zip")):
                continue
            if base.lower().endswith(".zip") and zf.getinfo(m).file_size > 2e9:
                continue                                   # большой вложенный архив — скорее всего фото
            target = os.path.join(DATA_DIR, base)
            if not os.path.exists(target):
                with zf.open(m) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                got.append(base)
                if base.lower().endswith(".zip"):
                    queue.append(target)
        if got:
            print(f"  из {os.path.basename(z)} распакованы: {', '.join(got)}")


def _key(name):
    """'Ad_table (extra).csv' / 'ad-table_extra.CSV' -> 'adtableextracsv' — сравнение без регистра и знаков."""
    return "".join(c for c in os.path.basename(name).lower() if c.isalnum())


def find_csv(word):
    w = _key(word)
    match = lambda: sorted((f for f in data_files("*.csv") if w in _key(f)),  # noqa: E731
                           key=lambda f: ("extra" in _key(f), len(f)))
    files = match()
    if not files:  # таблиц нет — может, они лежат в архиве (tables_V2.0.zip): достаём
        unpack_tables()
        files = match()
    return files[0] if files else None


def data_report():
    """Что лежит в data/ — печатается, когда чего-то не хватает."""
    lines = [f"  Папка с данными: {DATA_DIR}"]
    if not os.path.isdir(DATA_DIR):
        return "\n".join(lines + ["  [!] такой папки нет — создайте data/ рядом с train.py"])
    files = [os.path.join(r, f) for r, _, fs in os.walk(DATA_DIR) for f in fs
             if not (f.lower().endswith(IMG_EXT) and "$$" in f)]
    if not files:
        lines.append("  [!] папка пуста")
    for f in sorted(files):
        info = f"{os.path.getsize(f) / 1e9:.2f} ГБ"
        if f.lower().endswith((".crdownload", ".part", ".tmp", ".download")):
            info += "  <- НЕ ДОКАЧАН"
        elif f.lower().endswith(".zip"):
            try:
                names = zipfile.ZipFile(f).namelist()
                info += (f", внутри: CSV {sum(n.lower().endswith('.csv') for n in names)}, "
                         f"фото {sum(n.lower().endswith(IMG_EXT) for n in names)}")
            except (zipfile.BadZipFile, OSError):
                info += "  <- архив повреждён или не докачан"
        lines.append(f"    {os.path.relpath(f, ROOT)}  ({info})")
    return "\n".join(lines)


def text(s):
    """Как astype(str) в pandas 2: всё в строки, пропуски -> 'nan'. В pandas 3 astype(str) оставляет NaN,
    а CatBoost и кодировщики sklearn на NaN среди строк падают."""
    if isinstance(s, pd.DataFrame):
        return s.apply(text)
    return s.astype(object).where(s.notna(), "nan").astype(str).astype(object)


def col(df, *names):
    """Находит колонку по одному из возможных названий (без учёта регистра)."""
    low = {c.lower().strip(): c for c in df.columns}
    for n in names:
        if n.lower() in low:
            return df[low[n.lower()]]
    raise KeyError(f"Нет колонки {names}. Колонки файла: {list(df.columns)}")


def num(s):
    return pd.to_numeric(s.pipe(text).str.replace(r"[^0-9.]", "", regex=True), errors="coerce")


def load_ads():
    f = find_csv("ad_table")
    if not f:
        sys.exit("\n[ОШИБКА] Не найден Ad_table.csv — ни в data/, ни внутри zip-архивов в data/.\n"
                 + data_report() + "\n  Таблицы лежат в tables_V2.0.zip (0,2 ГБ) на странице датасета — положите его в data/.\n"
                 "  Архивы с фото (resized_DVM_v2.zip, Confirmed_fronts.zip) таблиц не содержат.")
    raw = pd.read_csv(f, low_memory=False)
    ads = pd.DataFrame({
        "maker": col(raw, "Maker").pipe(text).str.strip(),
        "model": (col(raw, "Maker").pipe(text).str.strip() + " " + col(raw, "Genmodel").pipe(text).str.strip()),
        "genmodel_id": col(raw, "Genmodel_ID").pipe(text),
        "adv_id": col(raw, "Adv_ID").pipe(text),
        "adv_year": num(col(raw, "Adv_year")),
        "reg_year": num(col(raw, "Reg_year")),
        "mileage": num(col(raw, "Runned_Miles", "Mileage")),
        "price": num(col(raw, "Price")),
        "color": col(raw, "Color", "Colour").pipe(text),
        "body": col(raw, "Bodytype", "Body_type").pipe(text),
        "gearbox": clean_cat(col(raw, "Gearbox")),
        "fuel": clean_cat(col(raw, "Fuel_type")),
        "engine_size": num(col(raw, "Engin_size", "Engine_size")),
    })
    ads["engine_size"] = ads.engine_size.where(ads.engine_size.between(0.6, 8.5))
    ads = ads.merge(load_tech(), on="adv_id", how="left")
    ads = ads.dropna(subset=["adv_year", "reg_year", "mileage", "price"])
    ads = ads[(ads.price.between(500, 250_000)) & (ads.mileage.between(0, 250_000)) &
              (ads.reg_year.between(PRICE_YEAR_MIN, PRICE_YEAR_MAX)) & (ads.adv_year >= ads.reg_year)]
    ads = ads.drop_duplicates("adv_id")
    ads = ads[ads.groupby("model")["model"].transform("size") >= 50]           # редкие модели убираем
    q = ads.groupby(["model", "reg_year"])["price"]                           # выбросы цены (опечатки)
    ads = ads[(ads.price >= q.transform(lambda s: s.quantile(.01))) & (ads.price <= q.transform(lambda s: s.quantile(.99)))].copy()
    # цена в сегодняшних фунтах
    ads["price_today"] = ads.price * ads.adv_year.clip(2015, 2026).astype(int).map(lambda y: UK_INDEX[2026] / UK_INDEX[y])
    ads["age"] = (ads.adv_year - ads.reg_year + 0.5).clip(lower=0.1)
    ads["split"] = ads.adv_id.map(split_of)
    return ads.reset_index(drop=True)


def clean_cat(s):
    s = s.pipe(text).str.strip().str.replace(r"\s+", " ", regex=True)
    return s.where(~s.isin(["nan", "", "None"]))


def load_tech():
    """Техданные из Ad_table_extra.csv (мощность, расход, габариты, места, двери). Нет файла — пустые колонки."""
    f = find_csv("ad_table_extra")
    if not f:
        print("  [!] Ad_table_extra.csv не найден — техданные (мощность, расход, габариты) не используются.\n"
              f"      Найденные таблицы: {[os.path.basename(x) for x in data_files('*.csv')]}")
        return pd.DataFrame(columns=["adv_id"] + SPEC_NUMS)
    raw = pd.read_csv(f, low_memory=False)
    t = pd.DataFrame({"adv_id": col(raw, "Adv_ID").pipe(text)})
    names = {"seats": "Seat_num", "doors": "Door_num", "engine_power": "Engine_power", "mpg": "Average_mpg",
             "top_speed": "Top_speed", "annual_tax": "Annual_Tax", "length": "Length", "width": "Width",
             "height": "Height", "wheelbase": "Wheelbase"}
    for k, c in names.items():
        t[k] = num(col(raw, c))
    return t.drop_duplicates("adv_id")


def split_of(adv_id):
    h = int(hashlib.md5(adv_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if h < 70 else "val" if h < 85 else "test"


IMG_EXT = (".jpg", ".jpeg", ".png")


def scan_photos():
    """Все фото DVM-CAR, которые есть на диске: в архивах *.zip (без распаковки) и в распакованных папках.
    Возвращает {имя_файла: (архив или None, путь)}."""
    where = {}
    zips = sorted(data_files("*.zip"), key=os.path.getsize, reverse=True)
    print(f"  архивы: {[os.path.basename(z) + f' ({os.path.getsize(z) / 1e9:.1f} ГБ)' for z in zips] or 'не найдены'}")
    for z in zips:
        try:
            members = zipfile.ZipFile(z).namelist()
        except zipfile.BadZipFile:
            print(f"  [!] архив {os.path.basename(z)} повреждён или не докачан — скачайте заново")
            continue
        n0 = len(where)
        for m in members:
            if m.lower().endswith(IMG_EXT):
                where.setdefault(os.path.basename(m), (z, m))
        print(f"    {os.path.basename(z)}: фото внутри — {len(where) - n0}")
    n0 = len(where)
    for root, _, files in os.walk(DATA_DIR):   # распакованные фото — только из data/
        for f in files:
            if f.lower().endswith(IMG_EXT) and "$$" in f:
                where.setdefault(f, (None, os.path.join(root, f)))
    if len(where) > n0:
        print(f"  в распакованных папках: {len(where) - n0} фото")
    print(f"  всего фото найдено: {len(where)}")
    if not where:
        print("\n[ОШИБКА] Не найдено ни одного фото. Положите архив с фото DVM-CAR (.zip) в папку data/:")
        print("  https://figshare.com/ndownloader/files/34792480  (фото спереди, 730 МБ)")
        print(data_report())
        sys.exit(1)
    return where


def load_images(ads, where):
    """Таблица фото по именам файлов: Марка$$Модель$$Год$$Цвет$$GenmodelID$$N$$image_k.jpg"""
    rows = []
    for name in where:
        p = name.split("$$")
        if len(p) >= 7:
            rows.append((name, f"{p[0].strip()} {p[1].strip()}", p[2], p[3], f"{p[4]}$${p[5]}"))
    if not rows:
        print("\n[ОШИБКА] Имена фото не в формате DVM-CAR. Пример:", list(where)[:3])
        sys.exit(1)
    df = pd.DataFrame(rows, columns=["name", "model_from_name", "year", "color_from_name", "adv_id"])
    df["year"] = pd.to_numeric(df.year, errors="coerce")
    # ракурс из Image_table (если файл есть); фото, которых нет в таблице, оставляем
    f = find_csv("image_table")
    if f:
        it = pd.read_csv(f, low_memory=False)
        views = dict(zip(col(it, "Image_name").pipe(text).map(os.path.basename), num(col(it, "Predicted_viewpoint"))))
        df["view"] = df.name.map(views)
        df = df[df.view.isna() | (df.view == VIEWPOINT)]
    df = df[df.year.between(YEAR_MIN, YEAR_MAX)]
    print(f"  фото {YEAR_MIN}–{YEAR_MAX} годов нужного ракурса: {len(df)}")
    # метки берём из объявления (если нашлось), иначе из имени файла
    df = df.merge(ads[["adv_id", "model", "color", "body", "split"]], on="adv_id", how="left")
    print(f"  из них сопоставлено с объявлениями: {df.model.notna().sum()}")
    df["model"] = df.model.fillna(df.model_from_name)
    df["color"] = df.color.fillna(df.color_from_name)
    df["body"] = df.body.fillna(df.model.map(ads.groupby("model").body.agg(lambda s: s.mode()[0])))
    df["split"] = df.split.fillna(df.adv_id.map(split_of))
    df = df[df.model.isin(set(ads.model)) & df.body.notna()]     # только модели, для которых есть цены
    df = df.sample(frac=1, random_state=SEED).drop_duplicates("adv_id")       # 1 фото на объявление
    return df.reset_index(drop=True)


def limit_images(df):
    """Модели с достаточным числом фото, не больше MAX_PER_MODEL на модель и MAX_IMAGES всего."""
    df = df[df.groupby("model")["model"].transform("size") >= MIN_PER_MODEL]
    df = df.groupby("model").head(MAX_PER_MODEL)
    if len(df) > MAX_IMAGES:
        # пропорционально урезаем каждую модель (без groupby.apply: в pandas 3 он теряет колонку model)
        quota = np.maximum(1, (df.groupby("model")["model"].transform("size") * MAX_IMAGES / len(df)).astype(int))
        df = df[df.groupby("model").cumcount() < quota]
    return df.reset_index(drop=True)


class PhotoStore:
    """Читает фото из архива (без распаковки) или из папки."""

    def __init__(self, where):
        self.where, self._open = where, {}

    def __call__(self, name):
        z, m = self.where[name]
        if z is None:
            return to_rgb(Image.open(m))
        key = (os.getpid(), z)
        if key not in self._open:
            self._open[key] = zipfile.ZipFile(z)
        with self._open[key].open(m) as f:
            return to_rgb(Image.open(f))

    def __getstate__(self):
        return {"where": self.where, "_open": {}}


def to_rgb(img):
    img.load()
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    return img.convert("RGB")


def load_params(key):
    import json
    f = path("best_params.json")
    return json.load(open(f, encoding="utf-8")).get(key) if os.path.exists(f) and not QUICK else None


def save_params(key, value):
    import json
    if QUICK:                                  # быстрый режим не портит подобранные параметры
        return
    f = path("best_params.json")
    d = json.load(open(f, encoding="utf-8")) if os.path.exists(f) else {}
    d[key] = value
    json.dump(d, open(f, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


# ------------------------------------------------------------------ 2. МОДЕЛЬ ЦЕНЫ
CATS = ["maker", "body", "color", "gearbox", "fuel"]
SPEC_KEYS = ["engine_size", "gearbox", "fuel"]          # это пользователь знает (или берём самое частое для модели)
SPEC_NUMS = ["seats", "doors", "engine_power", "mpg", "top_speed", "annual_tax", "length", "width", "height",
             "wheelbase"]                               # это подставляется из справочника Specs


class Specs:
    """Справочник техданных: медианы по train для «модель + объём + топливо + коробка» (с откатом на более общие
    уровни). При оценке пользователь знает только модель, объём, коробку и топливо — мощность, расход, габариты
    берутся отсюда. При обучении делаем так же, чтобы модель училась на том, что будет известно при оценке."""
    LEVELS = [["model", "engine_size", "fuel", "gearbox"], ["model", "engine_size", "fuel"],
              ["model", "engine_size"], ["model"]]

    def __init__(self, ads):
        mode = lambda s: s.mode().iloc[0] if s.notna().any() else np.nan  # noqa: E731
        self.defaults = ads.groupby("model")[SPEC_KEYS].agg(mode)
        self.engines = ads.dropna(subset=["engine_size"]).groupby("model").engine_size.agg(lambda s: sorted(set(s)))
        self.tables = [ads.groupby(k)[SPEC_NUMS].median() for k in self.LEVELS]

    def fill(self, df):
        """Пропуски в объёме/коробке/топливе → самые частые для модели; добавляет колонки SPEC_NUMS."""
        df = df.copy()
        for k in SPEC_KEYS:
            if k not in df:
                df[k] = np.nan
            df[k] = df[k].where(df[k].notna(), df.model.map(self.defaults[k]))
        df["engine_size"] = df.engine_size.astype(float)
        out = pd.DataFrame(np.nan, index=df.index, columns=SPEC_NUMS)
        for keys, t in zip(self.LEVELS, self.tables):
            idx = pd.MultiIndex.from_frame(df[keys]) if len(keys) > 1 else pd.Index(df[keys[0]])
            vals = t.reindex(idx)
            vals.index = df.index
            out = out.fillna(vals)
        df[SPEC_NUMS] = out
        return df


def new_price_table():
    f = find_csv("price_table")
    if not f:
        return {}
    t = pd.read_csv(f)
    t = pd.DataFrame({"gid": col(t, "Genmodel_ID").pipe(text), "year": num(col(t, "Year")),
                      "price": num(col(t, "Entry_price"))}).dropna()
    return {g: dict(zip(d.year.astype(int), d.price)) for g, d in t.groupby("gid")}


def new_price(table, gid, year):
    d = table.get(str(gid))
    if not d:
        return np.nan
    y = min(d, key=lambda k: abs(k - year))
    return d[y]


def price_features(df, table, specs):
    df = specs.fill(df)
    X = df[["model"] + CATS].copy()
    X["age"] = df["age"].values
    X["mileage"] = df["mileage"].values / 1000
    X["miles_per_year"] = df["mileage"].values / df["age"].values
    X["log_new_price"] = np.log([new_price(table, g, y) for g, y in zip(df.genmodel_id, df.reg_year)])
    X["engine_size"] = df.engine_size.values
    X[SPEC_NUMS] = df[SPEC_NUMS].values
    return X


class PriceModel:
    """CatBoost на логарифме цены. Категории (включая модель авто) кодирует сам CatBoost (ordered target
    statistics — без утечки). Ранняя остановка по 10 % train, чтобы val остался чистым для подбора и интервала."""
    CAT_COLS = ["model"] + CATS

    def __init__(self, params, iterations=None):
        self.params, self.iterations = params, iterations or PRICE_ITERS

    def _prep(self, X):
        return X.assign(**{c: text(X[c]) for c in self.CAT_COLS})

    def fit(self, X, y):
        global PRICE_GPU
        M = self._prep(X)
        rnd = np.random.RandomState(SEED).rand(len(M)) < 0.1      # 10 % train — для ранней остановки
        for gpu in ([True, False] if PRICE_GPU else [False]):
            device = dict(task_type="GPU", devices="0") if gpu else dict(task_type="CPU", thread_count=-1)
            self.cb = CatBoostRegressor(iterations=self.iterations, loss_function="RMSE", od_type="Iter", od_wait=100,
                                        bootstrap_type="Bayesian", random_seed=SEED, verbose=0,
                                        allow_writing_files=False, **device, **self.params)
            try:
                self.cb.fit(M[~rnd], y[~rnd], cat_features=self.CAT_COLS, eval_set=(M[rnd], y[rnd]),
                            use_best_model=True)
                return self
            except Exception as e:
                if not gpu:
                    raise
                print(f"  [!] CatBoost не смог обучаться на видеокарте ({str(e)[:120]}) — дальше на процессоре")
                PRICE_GPU = False
        return self

    def predict_log(self, X):
        return self.cb.predict(self._prep(X))

    def calibrate(self, X, y):
        """Интервал P10–P90 по ошибкам на валидации (конформный метод), отдельно для 5 ценовых групп."""
        p = self.predict_log(X)
        r = y - p
        self.edges = np.quantile(p, [.2, .4, .6, .8])
        g = np.digitize(p, self.edges)
        self.q = [(np.quantile(r[g == i], .1), np.quantile(r[g == i], .9)) for i in range(5)]

    def predict(self, X):
        p = self.predict_log(X)
        g = np.digitize(p, self.edges)
        lo = np.array([self.q[i][0] for i in g])
        hi = np.array([self.q[i][1] for i in g])
        return np.exp(p), np.exp(p + lo), np.exp(p + hi)


def mape(y, p):
    return float(np.mean(np.abs(p - y) / y))


def train_price(ads, table):
    print(f"  CatBoost обучается на: {'видеокарте (GPU)' if PRICE_GPU else 'процессоре (CPU)'}")
    tr, va, te = [(ads.split == s).values for s in ("train", "val", "test")]
    specs = Specs(ads[tr])                                  # справочник только по train — без утечки
    X, y = price_features(ads, table, specs), np.log(ads.price_today.values)
    params = dict(PRICE_PARAMS)
    saved = load_params("price_catboost")
    if saved:
        print(f"  подобранные ранее гиперпараметры CatBoost взяты из best_params.json "
              f"(удалите ключ price_catboost, чтобы подобрать заново)")
        params = saved
    elif TUNE:
        def objective(trial):
            t0 = time.time()
            p = dict(learning_rate=trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
                     depth=trial.suggest_int("depth", 5, 10),
                     l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 0.5, 30, log=True),
                     random_strength=trial.suggest_float("random_strength", 0.1, 10, log=True),
                     bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 2.0))
            m = PriceModel(p).fit(X[tr], y[tr])
            score = mape(np.exp(y[va]), np.exp(m.predict_log(X[va])))
            print(f"  попытка {trial.number + 1}/{PRICE_TRIALS}: MAPE val = {score:.4f} "
                  f"({m.cb.get_best_iteration() + 1} деревьев, {time.time() - t0:.0f} c) {p}")
            return score
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=SEED))
        study.enqueue_trial(PRICE_PARAMS)
        study.optimize(objective, n_trials=PRICE_TRIALS)
        params = study.best_params
        save_params("price_catboost", params)
    print(f"  гиперпараметры: {params}")
    model = PriceModel(params).fit(X[tr], y[tr])
    model.calibrate(X[va], y[va])
    res = {}
    for name, m in (("train", tr), ("val", va), ("test", te)):
        p, lo, hi = model.predict(X[m])
        yt = np.exp(y[m])
        res[name] = {"MAPE": mape(yt, p), "в пределах ±10%": float(np.mean(np.abs(p - yt) / yt <= .1)),
                     "попадание в интервал": float(np.mean((yt >= lo) & (yt <= hi)))}
    for k, v in res.items():
        print(f"  {k:<5}: MAPE {v['MAPE']:.1%}, в пределах ±10%: {v['в пределах ±10%']:.0%}, "
              f"в интервале: {v['попадание в интервал']:.0%}")
    return model, params, res, specs


# ------------------------------------------------------------------ 3. НЕЙРОСЕТЬ
def make_backbone(arch, pretrained):
    """Тело сети без классификатора -> (модуль, размер вектора признаков)."""
    if arch == "convnext_tiny":
        m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        return nn.Sequential(m.features, m.avgpool, m.classifier[0], m.classifier[1]), 768
    if arch == "efficientnet_v2_s":
        m = tvm.efficientnet_v2_s(weights=tvm.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None)
        return nn.Sequential(m.features, m.avgpool, nn.Flatten(1)), 1280
    if arch == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT if pretrained else None)
        m.fc = nn.Identity()
        return m, 2048
    raise ValueError(f"Неизвестная архитектура {arch}")


class CarNet(nn.Module):
    """Сеть (ConvNeXt-Tiny / EfficientNet-V2-S / ResNet-50) + 3 выхода: модель авто, цвет, кузов."""

    def __init__(self, n_model, n_color, n_body, dropout=0.2, pretrained=True, arch="convnext_tiny"):
        super().__init__()
        self.backbone, dim = make_backbone(arch, pretrained)
        self.drop = nn.Dropout(dropout)
        self.model_head, self.color_head, self.body_head = (nn.Linear(dim, n) for n in (n_model, n_color, n_body))

    def forward(self, x):
        f = self.drop(self.backbone(x))
        return self.model_head(f), self.color_head(f), self.body_head(f)


NORM = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def tf(train):
    if train:   # оттенок не меняем: цвет машины — это метка
        return T.Compose([T.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1)), T.RandomHorizontalFlip(),
                          T.ColorJitter(0.2, 0.2, 0.1), T.ToTensor(), NORM])
    return T.Compose([T.Resize(IMG_SIZE), T.CenterCrop(IMG_SIZE), T.ToTensor(), NORM])


class Photos(Dataset):
    def __init__(self, df, store, train):
        self.df, self.store, self.tf = df.reset_index(drop=True), store, tf(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        return self.tf(self.store(r["name"])), int(r.y_model), int(r.y_color), int(r.y_body)


def run_epochs(df_tr, df_va, store, n, params, epochs, arch, batch, trial=None):
    loader = lambda d, t: DataLoader(Photos(d, store, t), batch_size=batch, shuffle=t, num_workers=NUM_WORKERS,  # noqa
                                     pin_memory=DEVICE == "cuda", persistent_workers=NUM_WORKERS > 0,
                                     drop_last=t and len(d) > batch)
    dl_tr, dl_va = loader(df_tr, True), loader(df_va, False)
    net = CarNet(*n, params["dropout"], PRETRAINED, arch).to(DEVICE)
    heads = [p for name, p in net.named_parameters() if "head" in name]
    body = [p for name, p in net.named_parameters() if "head" not in name]
    opt = torch.optim.AdamW([{"params": body, "lr": params["lr"]}, {"params": heads, "lr": params["lr_head"]}],
                            weight_decay=params["weight_decay"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[params["lr"], params["lr_head"]],
                                                total_steps=epochs * len(dl_tr), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")
    best, best_state, bad, hist = -1, None, 0, []
    for ep in range(1, epochs + 1):
        t0, total = time.time(), 0.0
        net.train()
        for x, ym, yc, yb in dl_tr:
            x, ym, yc, yb = x.to(DEVICE), ym.to(DEVICE), yc.to(DEVICE), yb.to(DEVICE)
            with torch.autocast("cuda", enabled=DEVICE == "cuda"):
                om, oc, ob = net(x)
                ls = params["label_smoothing"]
                loss = (F.cross_entropy(om, ym, label_smoothing=ls) + 0.5 * F.cross_entropy(oc, yc, label_smoothing=ls)
                        + 0.5 * F.cross_entropy(ob, yb, label_smoothing=ls))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(x)
        acc = accuracy(net, dl_va)
        hist.append({"epoch": ep, "loss": total / len(df_tr), **acc})
        print(f"    эпоха {ep}/{epochs}: потери {total / len(df_tr):.3f} | точность val: модель {acc['model']:.3f}, "
              f"цвет {acc['color']:.3f}, кузов {acc['body']:.3f} | {time.time() - t0:.0f} c")
        if acc["model"] > best:
            best, bad = acc["model"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if trial is not None:
            trial.report(acc["model"], ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if bad >= PATIENCE:
            print("    ранняя остановка")
            break
    net.load_state_dict(best_state)
    return net, best, hist


@torch.no_grad()
def accuracy(net, loader):
    net.eval()
    hit, n = np.zeros(3), 0
    for x, ym, yc, yb in loader:
        with torch.autocast("cuda", enabled=DEVICE == "cuda"):
            outs = net(x.to(DEVICE))
        for i, (o, y) in enumerate(zip(outs, (ym, yc, yb))):
            hit[i] += (o.argmax(1).cpu() == y).sum().item()
        n += len(x)
    return dict(zip(("model", "color", "body"), hit / max(n, 1)))


@torch.no_grad()
def predict_proba(net, df, store):
    """Вероятности классов одной сети: [модель, цвет, кузов], каждая — массив (фото × классы)."""
    net.eval()
    dl = DataLoader(Photos(df, store, False), batch_size=BATCH, num_workers=NUM_WORKERS)
    out = [[], [], []]
    for x, *_ in dl:
        with torch.autocast("cuda", enabled=DEVICE == "cuda"):
            for i, o in enumerate(net(x.to(DEVICE))):
                out[i].append(torch.softmax(o.float(), 1).cpu().numpy())
    return [np.concatenate(o) for o in out]


def predict_all(nets, df, store):
    """Ответ ансамбля: среднее вероятностей всех сетей -> номера классов [модель, цвет, кузов]."""
    probs = [predict_proba(net, df, store) for net in nets]
    return [np.mean([p[i] for p in probs], axis=0).argmax(1) for i in range(3)], probs


def tune_cnn(arch, batch, tr, va, store, n):
    """Optuna для одной сети ансамбля: 3 эпохи на 30 % train, лучшая точность модели на val."""
    saved = load_params("cnn_" + arch)
    if saved:
        print(f"  [{arch}] подобранные ранее гиперпараметры взяты из best_params.json")
        return saved
    if not TUNE:
        return dict(CNN_PARAMS)

    def objective(trial):
        p = dict(lr=trial.suggest_float("lr", 3e-5, 5e-4, log=True),
                 lr_head=trial.suggest_float("lr_head", 3e-4, 5e-3, log=True),
                 weight_decay=trial.suggest_float("weight_decay", 0.01, 0.2, log=True),
                 label_smoothing=trial.suggest_float("label_smoothing", 0.0, 0.2),
                 dropout=trial.suggest_float("dropout", 0.0, 0.4))
        print(f"  [{arch}] попытка {trial.number + 1}/{CNN_TRIALS}: " + ", ".join(f"{k}={v:.3g}" for k, v in p.items()))
        _, best, _ = run_epochs(tr.sample(frac=0.3, random_state=trial.number), va.sample(frac=0.5, random_state=0),
                                store, n, p, 3, arch, batch, trial)
        return best
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=3))
    study.enqueue_trial(CNN_PARAMS)
    study.optimize(objective, n_trials=CNN_TRIALS)
    save_params("cnn_" + arch, study.best_params)
    return study.best_params


def train_cnn(imgs, store):
    """Ансамбль: для каждой архитектуры из ENSEMBLE — подбор гиперпараметров и финальное обучение."""
    classes = {k: sorted(imgs[k].unique()) for k in ("model", "color", "body")}
    for k in classes:
        imgs["y_" + k] = imgs[k].map({c: i for i, c in enumerate(classes[k])})
    n = [len(classes[k]) for k in ("model", "color", "body")]
    tr, va = imgs[imgs.split == "train"], imgs[imgs.split == "val"]
    members = []
    for i, (arch, batch) in enumerate(ENSEMBLE.items(), 1):
        print(f"\n  --- сеть {i}/{len(ENSEMBLE)}: {arch}")
        params = tune_cnn(arch, batch, tr, va, store, n)
        print(f"  [{arch}] финальное обучение, гиперпараметры: {params}")
        net, best, hist = run_epochs(tr, va, store, n, params, EPOCHS, arch, batch)
        print(f"  [{arch}] лучшая точность модели на val: {best:.3f}")
        members.append({"arch": arch, "net": net, "params": params, "hist": hist, "val": best})
    return members, classes


# ------------------------------------------------------------------ 4. ЗАПУСК
def price_report(params, res):
    gap = res["test"]["MAPE"] - res["train"]["MAPE"]
    lines = ["МОДЕЛЬ ЦЕНЫ (CatBoost)", f"признаки: модель, {', '.join(CATS)}, возраст, пробег, цена новой, объём, "
             f"{', '.join(SPEC_NUMS)} (по справочнику)", f"гиперпараметры: {params}"]
    lines += [f"  {k}: " + ", ".join(f"{a} {b:.1%}" for a, b in v.items()) for k, v in res.items()]
    lines.append(f"  разрыв MAPE train→test: {gap * 100:.1f} п.п. → "
                 + ("переобучения нет" if gap < 0.03 else "небольшой" if gap < 0.06 else "есть переобучение"))
    return lines


def save_price(ads, table, price_model, specs, res, e2e, recognition, n_images):
    market = ads[["model", "maker", "genmodel_id", "reg_year", "adv_year", "age", "mileage", "price_today", "color",
                  "body", "gearbox", "fuel", "engine_size"]]
    sales_f = find_csv("sales_table")
    sales = None
    if sales_f:
        s = pd.read_csv(sales_f)
        years = [c for c in s.columns if str(c).strip().isdigit()]
        sales = s.assign(gid=col(s, "Genmodel_ID").pipe(text)).melt("gid", years, "year", "sales")
        sales["year"] = sales.year.astype(int)
    joblib.dump({"model": price_model, "specs": specs, "new_prices": table, "market": market, "sales": sales,
                 "quality": {"price_test_mape": res["test"]["MAPE"], "e2e_mape": e2e, "recognition": recognition,
                             "coverage": res["test"]["попадание в интервал"], "n_ads": len(ads),
                             "n_images": n_images}}, path("price_model.joblib"))


def main_price_only(t0):
    """python train.py --price-only — переобучить только модель цены (фото не нужны; видеокарта — если есть).
    Нейросеть car_model.pt не трогается; её точность берётся из прежнего price_model.joblib."""
    print("1) Данные")
    ads = load_ads()
    print(f"  объявлений: {len(ads)}, моделей: {ads.model.nunique()}")
    print("\n2) Модель цены")
    table = new_price_table()
    price_model, params, res, specs = train_price(ads, table)
    save_price(ads, table, price_model, specs, res, e2e=None, **old_cnn_quality())
    report = [f"Модель цены обучена {time.strftime('%Y-%m-%d %H:%M')} за {(time.time() - t0) / 60:.0f} мин "
              f"(режим --price-only, нейросеть не переобучалась)", f"Объявлений: {len(ads)}", ""]
    report += price_report(params, res)
    report += ["", "Сквозная ошибка «цена по фото» не измерялась: нужно полное обучение (python train.py)."]
    open(path("metrics_price.txt"), "w", encoding="utf-8").write("\n".join(report))
    print(f"\nГотово за {(time.time() - t0) / 60:.0f} мин. Файлы: price_model.joblib, metrics_price.txt")


def old_cnn_quality():
    """Точность нейросети из прежнего price_model.joblib, а если он не читается (другая версия pandas) — из metrics.txt."""
    try:
        q = joblib.load(path("price_model.joblib"))["quality"]
        return {"recognition": q.get("recognition"), "n_images": q.get("n_images")}
    except Exception:
        import re
        txt = open(path("metrics.txt"), encoding="utf-8").read() if os.path.exists(path("metrics.txt")) else ""
        r, n_img = re.search(r"на тесте: модель ([\d.]+)%", txt), re.search(r"фото: (\d+)", txt)
        return {"recognition": float(r.group(1)) / 100 if r else None, "n_images": int(n_img.group(1)) if n_img else None}


def main():
    t0 = time.time()
    if "--price-only" in sys.argv:
        return main_price_only(t0)
    print(f"Устройство: {DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else ""))
    if DEVICE == "cpu" and not QUICK and "--cpu" not in sys.argv:
        print(f"\n[ОШИБКА] PyTorch не видит видеокарту (установлен torch {torch.__version__}).")
        if "+cu" not in torch.__version__:
            print("  Установлена версия для процессора. Нужна версия с CUDA (в названии +cu126 / +cu128).")
        else:
            print("  Версия с CUDA установлена, но видеокарта не найдена: обновите драйвер NVIDIA и перезагрузите ПК.")
        print("  Обучение нейросети на процессоре займёт несколько суток.")
        print("  Если всё равно хотите на процессоре: python train.py --cpu")
        sys.exit(1)
    print("\n1) Данные")
    ads = load_ads()
    print(f"  объявлений: {len(ads)}, моделей: {ads.model.nunique()}")
    where = scan_photos()
    imgs = limit_images(load_images(ads, where))
    store = PhotoStore({n: where[n] for n in imgs.name})
    if len(imgs) == 0:
        sys.exit(f"[ОШИБКА] Фото найдены, но ни у одной модели нет {MIN_PER_MODEL}+ фото. "
                 f"Уменьшите MIN_PER_MODEL в начале train.py.")
    print(f"  отобрано фото: {len(imgs)} ({imgs.model.nunique()} моделей), train/val/test: "
          f"{(imgs.split == 'train').sum()}/{(imgs.split == 'val').sum()}/{(imgs.split == 'test').sum()}")

    print("\n2) Модель цены")
    table = new_price_table()
    price_model, price_params, price_res, specs = train_price(ads, table)

    print("\n3) Нейросеть распознавания (ансамбль)")
    members, classes = train_cnn(imgs, store)
    nets = [m["net"] for m in members]

    print("\n4) Проверка на тесте")
    test = imgs[imgs.split == "test"].reset_index(drop=True)
    (pm, pc, pb), probs = predict_all(nets, test, store)
    rec = {"модель": float(np.mean(pm == test.y_model)), "цвет": float(np.mean(pc == test.y_color)),
           "кузов": float(np.mean(pb == test.y_body))}
    rec_members = {m["arch"]: float(np.mean(p[0].argmax(1) == test.y_model)) for m, p in zip(members, probs)}
    trn = imgs[imgs.split == "train"].sample(min(3000, (imgs.split == "train").sum()), random_state=0)
    rec_train = float(np.mean(predict_all(nets, trn, store)[0][0] == trn.y_model.values))
    print("  точность модели по отдельным сетям: " + ", ".join(f"{a} {v:.1%}" for a, v in rec_members.items()))
    print(f"  точность ансамбля на тесте: " + ", ".join(f"{k} {v:.1%}" for k, v in rec.items())
          + f" (модель на train: {rec_train:.1%})")
    # сквозная проверка: модель/цвет/кузов из фото + настоящий год и пробег
    test["model"] = np.array(classes["model"])[pm]
    test["color"] = np.array(classes["color"])[pc]
    test["body"] = np.array(classes["body"])[pb]
    t = test[["adv_id", "model", "color", "body"]].merge(
        ads.drop(columns=["model", "color", "body", "split"]), on="adv_id")   # только фото с известной ценой
    t["genmodel_id"] = t.model.map(ads.groupby("model").genmodel_id.agg(lambda s: s.mode()[0]))
    t["maker"] = t.model.map(ads.groupby("model").maker.first())
    # объём/коробка/топливо — из объявления (их пользователь укажет сам), остальное — по справочнику
    e2e = mape(t.price_today.values, price_model.predict(price_features(t, table, specs))[0])
    print(f"  ошибка цены по фото (распознавание + год + пробег + двигатель): MAPE {e2e:.1%}")

    # ---- сохранение
    torch.save({"members": [{"arch": m["arch"], "weights": m["net"].state_dict(), "params": m["params"]}
                            for m in members], "classes": classes, "img_size": IMG_SIZE}, path("car_model.pt"))
    save_price(ads, table, price_model, specs, price_res, e2e, rec["модель"], len(imgs))
    report = [f"Обучено {time.strftime('%Y-%m-%d %H:%M')} за {(time.time() - t0) / 60:.0f} мин",
              f"Объявлений: {len(ads)}, фото: {len(imgs)}, моделей на фото: {len(classes['model'])}", ""]
    report += price_report(price_params, price_res)
    report += ["", "НЕЙРОСЕТЬ (ансамбль: " + ", ".join(ENSEMBLE) + ")"]
    report += [f"  {m['arch']}: гиперпараметры {m['params']}, точность модели на тесте {rec_members[m['arch']]:.1%}"
               for m in members]
    report += ["  точность ансамбля на тесте: " + ", ".join(f"{k} {v:.1%}" for k, v in rec.items()),
               f"  точность модели на train: {rec_train:.1%} (разрыв с тестом {(rec_train - rec['модель']) * 100:.1f} п.п.)",
               "", f"ЦЕНА ПО ФОТО (тест): MAPE {e2e:.1%}"]
    open(path("metrics.txt"), "w", encoding="utf-8").write("\n".join(report))
    plot_curves(members, price_res)
    print(f"\nГотово за {(time.time() - t0) / 60:.0f} мин. Файлы: car_model.pt, price_model.joblib, metrics.txt, "
          f"training_curves.png")


def plot_curves(members, price_res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4))
    for m in members:
        a.plot([h["epoch"] for h in m["hist"]], [h["model"] for h in m["hist"]], "o-", label=m["arch"])
    a.set_xlabel("эпоха"); a.set_ylabel("точность модели (val)"); a.set_title("Обучение сетей ансамбля")
    a.legend(); a.grid(alpha=.3)
    names = list(price_res)
    b.bar(names, [price_res[k]["MAPE"] * 100 for k in names], color=["#9aa3b2", "#9aa3b2", "#1f4fb4"])
    b.set_ylabel("MAPE, %"); b.set_title("Ошибка модели цены: train / val / test")
    fig.tight_layout()
    fig.savefig(path("training_curves.png"), dpi=130)


if __name__ == "__main__":
    main()
