ОЦЕНКА ЦЕНЫ АВТОМОБИЛЯ ПО ФОТО (DVM-CAR)

Структура:
  train.py           обучение: модель цены (CatBoost) + ансамбль из 3 нейросетей
                     (ConvNeXt-Tiny, EfficientNet-V2-S, ResNet-50), подбор гиперпараметров Optuna
  predict.py         фото + год + пробег (+ объём, коробка, топливо) -> цена в рублях + PDF-отчёт
  compare_prices.py  сравнение моделей цены (HistGB / Random Forest / CatBoost)
  requirements.txt   библиотеки
  data/              <-- СЮДА КЛАДЁТСЯ ДАТАСЕТ (что именно — в data/PUT_DATASET_HERE.txt)

1. Установка (один раз):
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   pip install -r requirements.txt

2. Датасет DVM-CAR -> в папку data/
   https://figshare.com/articles/figure/DVM-CAR_Dataset/19586296/1
     tables_V2.0.zip       (таблицы, можно не распаковывать)
     Confirmed_fronts.zip  (фото спереди, не распаковывать)

3. Обучение (RTX 4070: примерно 3-5 часов вместе с подбором гиперпараметров):
   Всё обучается на видеокарте: нейросети (PyTorch CUDA, смешанная точность) и CatBoost (task_type=GPU).
   В начале лога должно быть «Устройство: cuda (NVIDIA ...)» и «CatBoost обучается на: видеокарте (GPU)».
   Если CatBoost не найдёт видеокарту, он сам перейдёт на процессор и напишет об этом.
   python train.py
   Только модель цены (без фото):               python train.py --price-only
   Проверка, что всё запускается (минуты):       python train.py --quick

   Гиперпараметры, подобранные один раз, сохраняются в best_params.json и при следующих
   запусках берутся оттуда. Удалите файл, чтобы подобрать заново.

4. Оценка:
   python predict.py фото.jpg 2017 85000 2.0 автомат дизель
   (объём, коробка, топливо — по желанию; без них берутся самые частые для модели, точность ниже)

После обучения в папке проекта появятся:
   car_model.pt, price_model.joblib, best_params.json, metrics.txt, training_curves.png
