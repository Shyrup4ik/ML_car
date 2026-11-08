"""
СРАВНЕНИЕ МОДЕЛЕЙ ЦЕНЫ: текущий бустинг (sklearn) / Random Forest / CatBoost
на трёх наборах признаков. Фото не нужны — только CSV из папки.

Запуск:  python compare_prices.py          (≈10–20 мин на ноутбуке)
         python compare_prices.py --quick  (20 % данных, проверка кода)
Результат: price_comparison.txt и price_comparison.csv

Разбиение train/val/test, очистка данных и приведение цен к 2026 г. — те же, что в train.py.
Тест не участвует ни в обучении, ни в ранней остановке: на нём только итоговые цифры.
"""
import sys
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder

from train import SEED, load_ads, load_params, mape, new_price, new_price_table, path

QUICK = "--quick" in sys.argv

# ------------------------------------------------------------------ наборы признаков
BASE_CATS = ["maker", "body", "color"]
# что знает любой владелец (можно спросить в predict.py)
ENGINE_CATS, ENGINE_NUMS = ["gearbox", "fuel"], ["engine_size", "seats", "doors"]
# техданные модели из Ad_table_extra.csv (есть не у всех объявлений, пропуски — NaN)
FULL_NUMS = ["engine_power", "mpg", "top_speed", "annual_tax", "length", "width", "height", "wheelbase"]

SETS = {
    "базовые (как сейчас)": (BASE_CATS, []),
    "+ мотор/коробка/топливо": (BASE_CATS + ENGINE_CATS, ENGINE_NUMS),
    "+ все техданные": (BASE_CATS + ENGINE_CATS, ENGINE_NUMS + FULL_NUMS),
}


def load_data():
    ads = load_ads()                               # train.load_ads уже читает мотор/коробку/топливо и техданные
    if QUICK:
        ads = ads.sample(frac=0.2, random_state=SEED).reset_index(drop=True)
    return ads


def features(ads, table, cats, nums):
    """Признаки из самих объявлений (реальные техданные каждой машины, а не справочник, как в train.py) —
    это верхняя граница: при оценке мощность и расход конкретной комплектации неизвестны."""
    X = ads[["model"] + cats].copy()
    X["age"] = ads.age.values
    X["mileage"] = ads.mileage.values / 1000
    X["miles_per_year"] = ads.mileage.values / ads.age.values
    X["log_new_price"] = np.log([new_price(table, g, y) for g, y in zip(ads.genmodel_id, ads.reg_year)])
    for c in nums:
        X[c] = ads[c].values
    return X


# ------------------------------------------------------------------ модели
class Encoded:
    """Кодирование как в train.PriceModel: модель авто — target encoding (5 fold), остальные категории — порядковые."""

    def __init__(self, cats):
        self.cats = cats
        self.te = TargetEncoder(target_type="continuous", cv=KFold(5, shuffle=True, random_state=SEED))
        self.oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)

    def fit_transform(self, X, y):
        te = self.te.fit_transform(X[["model"]], y).ravel()
        self.oe.fit(X[self.cats].astype(str))
        return self._m(X, te)

    def transform(self, X):
        return self._m(X, self.te.transform(X[["model"]]).ravel())

    def _m(self, X, te):
        M = X.drop(columns=["model"]).copy()
        M[self.cats] = self.oe.transform(X[self.cats].astype(str))
        M.insert(0, "model_enc", te)
        return M


def fit_hgb(Xtr, ytr, Xva, yva, cats):
    """Текущая модель проекта: HistGradientBoosting с подобранными гиперпараметрами из best_params.json."""
    enc = Encoded(cats)
    M = enc.fit_transform(Xtr, ytr)
    p = dict(load_params("price") or dict(learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=50,
                                          l2_regularization=1.0, max_features=0.8))
    gbm = HistGradientBoostingRegressor(
        max_iter=3000, early_stopping=True, validation_fraction=0.1, n_iter_no_change=50,
        categorical_features=[c in cats for c in M.columns], random_state=SEED, **p)
    gbm.fit(M, ytr)
    return lambda X: gbm.predict(enc.transform(X))


def fit_rf(Xtr, ytr, Xva, yva, cats):
    """Random Forest (бэггинг). Пропуски в техданных sklearn ≥1.4 обрабатывает сам."""
    enc = Encoded(cats)
    M = enc.fit_transform(Xtr, ytr)
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.5, max_samples=0.7,
                               n_jobs=-1, random_state=SEED)
    rf.fit(M, ytr)
    return lambda X: rf.predict(enc.transform(X))


def fit_catboost(Xtr, ytr, Xva, yva, cats):
    """CatBoost: все категории (включая модель авто) подаются строками, кодирование — внутри CatBoost.
    Ранняя остановка по val (тест не трогаем)."""
    cat_cols = ["model"] + cats
    prep = lambda X: X.assign(**{c: X[c].astype(str) for c in cat_cols})  # noqa: E731
    cb = CatBoostRegressor(iterations=5000, learning_rate=0.08, depth=8, l2_leaf_reg=3, loss_function="RMSE",
                           od_type="Iter", od_wait=100, random_seed=SEED, verbose=0, thread_count=-1,
                           allow_writing_files=False)
    cb.fit(prep(Xtr), ytr, cat_features=cat_cols, eval_set=(prep(Xva), yva), use_best_model=True)
    return lambda X: cb.predict(prep(X))


MODELS = {"HistGB (текущая)": fit_hgb, "Random Forest": fit_rf, "CatBoost": fit_catboost}


# ------------------------------------------------------------------ запуск
def scores(y_log, p_log):
    y, p = np.exp(y_log), np.exp(p_log)
    e = np.abs(p - y) / y
    return {"MAPE": mape(y, p), "±10%": float(np.mean(e <= .1)), "медиана ошибки": float(np.median(e))}


def main():
    t_all = time.time()
    ads = load_data()
    table = new_price_table()
    y = np.log(ads.price_today.values)
    tr, va, te = [(ads.split == s).values for s in ("train", "val", "test")]
    print(f"Объявлений: {len(ads)} (train {tr.sum()}, val {va.sum()}, test {te.sum()})\n")

    rows = []
    for set_name, (cats, nums) in SETS.items():
        X = features(ads, table, cats, nums)
        print(f"== {set_name}: {X.shape[1]} признаков")
        preds = {}
        for model_name, fit in MODELS.items():
            t0 = time.time()
            predict = fit(X[tr], y[tr], X[va], y[va], cats)
            preds[model_name] = predict(X[te])
            s_tr = scores(y[tr], predict(X[tr]))
            s = scores(y[te], preds[model_name])
            dt = time.time() - t0
            rows.append({"признаки": set_name, "модель": model_name, "MAPE test": s["MAPE"],
                         "±10% test": s["±10%"], "медиана ошибки test": s["медиана ошибки"],
                         "MAPE train": s_tr["MAPE"], "время, с": dt})
            print(f"  {model_name:<18} MAPE test {s['MAPE']:.2%} | ±10%: {s['±10%']:.1%} | "
                  f"MAPE train {s_tr['MAPE']:.2%} | {dt:.0f} c")
        avg = (preds["HistGB (текущая)"] + preds["CatBoost"]) / 2           # простой ансамбль, без обучения
        s = scores(y[te], avg)
        rows.append({"признаки": set_name, "модель": "среднее HistGB+CatBoost", "MAPE test": s["MAPE"],
                     "±10% test": s["±10%"], "медиана ошибки test": s["медиана ошибки"],
                     "MAPE train": np.nan, "время, с": np.nan})
        print(f"  {'среднее HistGB+Cat':<18} MAPE test {s['MAPE']:.2%} | ±10%: {s['±10%']:.1%}\n")

    res = pd.DataFrame(rows)
    res.to_csv(path("price_comparison.csv"), index=False, encoding="utf-8")
    pivot = res.pivot(index="модель", columns="признаки", values="MAPE test")[list(SETS)]
    pivot = pivot.loc[list(MODELS) + ["среднее HistGB+CatBoost"]]
    best = res.loc[res["MAPE test"].idxmin()]
    base = res[(res.модель == "HistGB (текущая)") & (res.признаки == list(SETS)[0])].iloc[0]
    lines = [f"Сравнение моделей цены ({time.strftime('%Y-%m-%d %H:%M')}, {len(ads)} объявлений"
             + (", QUICK 20%" if QUICK else "") + ")",
             "", "MAPE на тесте (меньше — лучше):",
             pivot.map(lambda v: f"{v:.2%}").to_string(),
             "", "Доля оценок в пределах ±10% на тесте:",
             res.pivot(index="модель", columns="признаки", values="±10% test")[list(SETS)]
             .loc[pivot.index].map(lambda v: f"{v:.1%}").to_string(),
             "", f"Лучший вариант: {best.модель}, {best.признаки} — MAPE {best['MAPE test']:.2%} "
             f"(сейчас {base['MAPE test']:.2%}, выигрыш {(base['MAPE test'] - best['MAPE test']) * 100:.2f} п.п.)",
             f"Время: {(time.time() - t_all) / 60:.0f} мин"]
    open(path("price_comparison.txt"), "w", encoding="utf-8").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
