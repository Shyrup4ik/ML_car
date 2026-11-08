'use strict';

const $ = (id) => document.getElementById(id);
const state = { ready: false, busy: false, file: null, objectURL: null, metadata: null, recognition: null, session: null, result: null, resizeObserver: null };
const numberFormat = new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 0 });
const decimalFormat = new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 1 });
const smallFormat = new Intl.NumberFormat('ru-RU', { notation: 'compact', maximumFractionDigits: 1 });
const number = (value) => value == null ? "—" : numberFormat.format(value);
const metricPercent = (value) => value == null ? "не измерялась" : decimalFormat.format(value * 100) + "%";
const rub = (value, rounded = true) => number(rounded ? Math.round(value / 10000) * 10000 : value) + '\u00a0₽';
const gbp = (value) => '£' + number(value);
const signed = (value, suffix = '') => (value > 0 ? '+' : value < 0 ? '−' : '') + decimalFormat.format(Math.abs(value)) + suffix;
const escapeHTML = (value) => String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
const icon = (name, extra = '') => `<svg class="icon ${extra}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
const reducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;
function scrollToElement(element) { element.scrollIntoView({ behavior: reducedMotion() ? 'auto' : 'smooth', block: 'start' }); }

async function api(path, options = {}, timeout = 120000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(path, { ...options, signal: controller.signal });
    let data;
    try { data = await response.json(); } catch (_) { data = {}; }
    if (!response.ok) {
      const error = new Error(typeof data.detail === 'string' ? data.detail : 'Не удалось выполнить запрос. Попробуй ещё раз.');
      error.status = response.status;
      throw error;
    }
    return data;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('Обработка заняла слишком много времени. Попробуй ещё раз или выбери другое фото.');
    if (error instanceof TypeError) throw new Error('Не удалось связаться с приложением. Проверь, что локальный сервер запущен.');
    throw error;
  } finally { clearTimeout(timer); }
}

function setServerStatus(status, message) {
  $('server-status').dataset.state = status;
  $('server-status-text').textContent = message;
}

async function initialize() {
  try {
    const health = await api('/api/health', {}, 15000);
    setServerStatus(health.status, health.message);
    if (health.status === 'loading') {
      $('form-status').textContent = 'Первый запуск: подготавливаем модели. Уже можно выбрать фото и заполнить данные.';
      setTimeout(initialize, 1800);
      return;
    }
    if (health.status === 'error') {
      setFormStatus('Не удалось загрузить модели. Перезапусти приложение из папки проекта.', 'error');
      return;
    }
    state.metadata = await api('/api/metadata', {}, 20000);
    state.ready = true;
    populateOptions();
    $('stat-ads').textContent = number(state.metadata.quality.n_ads);
    $('stat-images').textContent = number(state.metadata.quality.n_images);
    $('stat-models').textContent = number(state.metadata.recognition_models);
    if (!state.busy) setFormStatus('Фото обрабатывается на твоём компьютере.');
  } catch (error) {
    setServerStatus('error', 'Нет соединения');
    setFormStatus(error.message, 'error');
    setTimeout(initialize, 5000);
  }
}

function populateOptions() {
  $('model-options').replaceChildren(...state.metadata.models.map((model) => {
    const option = document.createElement('option'); option.value = model; return option;
  }));
  for (const [id, entries] of [['color-select', state.metadata.colors], ['body-select', state.metadata.bodies]]) {
    $(id).replaceChildren(...entries.map((entry) => {
      const option = document.createElement('option'); option.value = entry.value; option.textContent = entry.label; return option;
    }));
  }
}

function setFormStatus(message, kind = '') {
  $('form-status').textContent = message;
  $('form-status').className = 'form-status' + (kind ? ' is-' + kind : '');
}

function setBusy(busy, action = 'recognize') {
  state.busy = busy;
  ['photo-input', 'clear-photo', 'paste-photo', 'year-input', 'mileage-input', 'sample-button', 'model-input', 'color-select', 'body-select', 'recognize-button', 'estimate-button'].forEach((id) => { $(id).disabled = busy; });
  $('vehicle-form').setAttribute('aria-busy', String(busy));
  const button = $(action === 'estimate' ? 'estimate-button' : 'recognize-button');
  if (busy) {
    button.innerHTML = `<span class="spinner" aria-hidden="true"></span><span>${action === 'estimate' ? 'Считаем стоимость…' : 'Распознаём автомобиль…'}</span>`;
  } else {
    $('recognize-button').innerHTML = `<span>${state.recognition ? 'Проверить и рассчитать' : 'Распознать автомобиль'}</span>${icon('arrow')}`;
    $('estimate-button').innerHTML = `<span>${state.result ? 'Обновить оценку' : 'Рассчитать стоимость'}</span>${icon('arrow')}`;
  }
}

function discardSession() {
  if (state.session) fetch('/api/sessions/' + encodeURIComponent(state.session), { method: 'DELETE', keepalive: true }).catch(() => {});
  state.session = null;
}

function clearResults() {
  state.result = null;
  $('results').hidden = true;
  $('results').replaceChildren();
  if (state.resizeObserver) state.resizeObserver.disconnect();
  state.resizeObserver = null;
  $('chart-tooltip').hidden = true;
}

function selectPhoto(file) {
  if (state.busy || !file) return false;
  if (file.size > 15 * 1024 * 1024) { $('photo-error').textContent = 'Фотография должна быть не больше 15 МБ.'; return false; }
  const accepted = ['image/jpeg', 'image/png', 'image/webp'];
  if (!accepted.includes(file.type) && !(file.type === '' && /\.(jpe?g|png|webp)$/i.test(file.name))) {
    $('photo-error').textContent = 'Выбери фото в формате JPG, PNG или WebP. HEIC можно предварительно сохранить как JPG.';
    return false;
  }
  discardSession();
  state.recognition = null;
  clearResults();
  $('recognition-panel').hidden = true;
  if (state.objectURL) URL.revokeObjectURL(state.objectURL);
  state.file = file;
  state.objectURL = URL.createObjectURL(file);
  $('preview-image').src = state.objectURL;
  $('photo-preview').hidden = false;
  $('clear-photo').hidden = false;
  $('upload-placeholder').hidden = true;
  $('dropzone').classList.add('has-photo');
  $('photo-error').textContent = '';
  $('review-error').textContent = '';
  setBusy(false);
  setFormStatus(state.ready ? 'Фото обрабатывается на твоём компьютере.' : 'Подготавливаем модели. Фото уже загружено.');
  return true;
}

function acceptPastedPhoto(file) {
  if (!selectPhoto(file)) return false;
  setFormStatus('Фото вставлено. Проверь год выпуска и пробег.');
  scrollToElement($('estimate'));
  return true;
}

function handlePhotoPaste(event) {
  if (!event.clipboardData) return;
  const imageItem = Array.from(event.clipboardData.items || []).find((item) => item.kind === 'file' && item.type.startsWith('image/'));
  const file = imageItem ? imageItem.getAsFile() : Array.from(event.clipboardData.files || []).find((entry) => entry.type.startsWith('image/') || /\.(jpe?g|png|webp)$/i.test(entry.name));
  // Text, including pasted year/mileage/model names, keeps the browser's normal behavior.
  if (!file) return;
  event.preventDefault();
  if (state.busy) { setFormStatus('Дождись завершения текущего расчёта, затем вставь новое фото.', 'loading'); return; }
  acceptPastedPhoto(file);
}

let readingClipboard = false;
async function pastePhotoFromClipboard() {
  if (state.busy || readingClipboard) return;
  if (!navigator.clipboard || typeof navigator.clipboard.read !== 'function') {
    $('photo-error').textContent = 'Скопируй само изображение, затем нажми ⌘V на Mac или Ctrl+V на Windows прямо на этой странице.';
    return;
  }
  const previousPhoto = state.file;
  readingClipboard = true;
  const button = $('paste-photo');
  button.disabled = true;
  button.setAttribute('aria-busy', 'true');
  button.innerHTML = '<span class="spinner" aria-hidden="true"></span><span>Вставляем…</span>';
  try {
    // Read only on this explicit button click. The page never inspects the clipboard on load.
    const entries = await navigator.clipboard.read();
    const imageEntry = entries.find((entry) => entry.types.some((type) => type.startsWith('image/')));
    if (!imageEntry) {
      $('photo-error').textContent = 'В буфере нет картинки. В браузере выбери «Копировать изображение», затем вставь его сюда.';
      return;
    }
    const type = ['image/png', 'image/jpeg', 'image/webp'].find((candidate) => imageEntry.types.includes(candidate)) || imageEntry.types.find((candidate) => candidate.startsWith('image/'));
    const blob = await imageEntry.getType(type);
    if (state.busy || state.file !== previousPhoto) return;
    const extension = { 'image/png': 'png', 'image/jpeg': 'jpg', 'image/webp': 'webp' }[type] || 'image';
    acceptPastedPhoto(new File([blob], 'вставленное-фото.' + extension, { type }));
  } catch (error) {
    $('photo-error').textContent = error.name === 'NotAllowedError' || error.name === 'SecurityError'
      ? 'Браузер не разрешил вставку кнопкой. Скопируй изображение и нажми ⌘V или Ctrl+V прямо на странице.'
      : 'Не удалось прочитать картинку из буфера. Попробуй вставить её через ⌘V или Ctrl+V.';
  } finally {
    readingClipboard = false;
    button.disabled = state.busy;
    button.removeAttribute('aria-busy');
    button.innerHTML = icon('clipboard') + '<span>Вставить фото</span>';
  }
}

function clearPhoto() {
  if (state.busy) return;
  discardSession();
  if (state.objectURL) URL.revokeObjectURL(state.objectURL);
  Object.assign(state, { file: null, objectURL: null, recognition: null });
  $('photo-input').value = '';
  $('preview-image').removeAttribute('src');
  $('photo-preview').hidden = true;
  $('clear-photo').hidden = true;
  $('upload-placeholder').hidden = false;
  $('dropzone').classList.remove('has-photo');
  $('recognition-panel').hidden = true;
  $('prepared-image').removeAttribute('src');
  $('photo-error').textContent = '';
  clearResults();
  setBusy(false);
}

function validateFields() {
  const currentYear = state.metadata ? state.metadata.current_year : new Date().getFullYear();
  const yearText = $('year-input').value.trim();
  const mileageText = $('mileage-input').value.replace(/[\s\u00a0\u202f]/g, '').replace(',', '.');
  let valid = true;
  const yearValid = /^\d{4}$/.test(yearText) && Number(yearText) >= 1990 && Number(yearText) <= currentYear;
  const mileageValid = /^\d+(\.\d{1,2})?$/.test(mileageText) && Number(mileageText) >= 0 && Number(mileageText) <= 1000000;
  for (const [name, ok, message] of [
    ['year', yearValid, `Укажи год от 1990 до ${currentYear}.`],
    ['mileage', mileageValid, 'Укажи пробег от 0 до 1 000 000 км.'],
  ]) {
    $(name + '-error').textContent = ok ? '' : message;
    $(name + '-input').setAttribute('aria-invalid', String(!ok));
    if (!ok) valid = false;
  }
  if (!valid) { $(yearValid ? 'mileage-input' : 'year-input').focus(); return null; }
  return { year: Number(yearText), mileage_km: Number(mileageText) };
}

async function recognize(event) {
  event.preventDefault();
  if (state.busy) return;
  if (!state.file) { $('photo-error').textContent = 'Сначала выбери фотографию автомобиля.'; $('photo-input').focus(); return; }
  const fields = validateFields();
  if (!fields) return;
  if (!state.ready) { setFormStatus('Модели ещё подготавливаются. Подожди немного и нажми кнопку снова.', 'loading'); return; }
  if (state.recognition) { scrollToElement($('recognition-panel')); $('model-input').focus({ preventScroll: true }); return; }
  setBusy(true);
  setFormStatus('Убираем фон и определяем автомобиль. Первый расчёт может занять немного больше времени.', 'loading');
  $('photo-error').textContent = '';
  try {
    const form = new FormData(); form.append('photo', state.file, state.file.name);
    const data = await api('/api/recognize', { method: 'POST', body: form }, 180000);
    state.recognition = data;
    state.session = data.session_id;
    renderRecognition(data);
    setFormStatus('Готово. Проверь распознавание ниже.');
    scrollToElement($('recognition-panel'));
  } catch (error) {
    setFormStatus(error.message, 'error');
    if (error.status === 422 || error.status === 413) $('photo-error').textContent = error.message;
  } finally { setBusy(false); }
}

function renderRecognition(data) {
  const recognized = data.recognition;
  $('model-input').value = recognized.model[0].value;
  $('color-select').value = recognized.color[0].value;
  $('body-select').value = recognized.body[0].value;
  $('prepared-image').src = data.prepared_image;
  const confidence = recognized.model[0].confidence;
  const low = confidence < .4;
  $('recognition-confidence').className = 'recognition-confidence' + (low ? ' low' : '');
  $('recognition-confidence').replaceChildren();
  const indicator = document.createElement('span'); indicator.className = 'confidence-indicator';
  $('recognition-confidence').append(indicator, document.createTextNode(`Уверенность распознавания: ${Math.round(confidence * 100)}%. ${low ? 'Результат неуверенный — обязательно проверь модель.' : 'Сверь название с автомобилем на фото.'}`));
  const alternatives = $('alternative-models'); alternatives.replaceChildren();
  if (recognized.model.length > 1) {
    const label = document.createElement('span'); label.textContent = 'Другие варианты:'; alternatives.append(label);
    recognized.model.slice(1).forEach((entry) => {
      const button = document.createElement('button'); button.type = 'button';
      button.textContent = entry.value + ' · ' + Math.round(entry.confidence * 100) + '%';
      button.addEventListener('click', () => { if (!state.busy) { $('model-input').value = entry.value; $('model-input').focus(); } });
      alternatives.append(button);
    });
  }
  $('review-error').textContent = '';
  $('recognition-panel').hidden = false;
}

async function estimate() {
  if (state.busy || !state.session) return;
  const fields = validateFields();
  if (!fields) { scrollToElement($('vehicle-form')); return; }
  const typed = $('model-input').value.trim();
  const model = state.metadata.models.find((candidate) => candidate.toLocaleLowerCase() === typed.toLocaleLowerCase());
  if (!model) { $('review-error').textContent = 'Выбери модель автомобиля из предложенного списка.'; $('model-input').focus(); return; }
  $('model-input').value = model;
  $('review-error').textContent = '';
  setBusy(true, 'estimate');
  try {
    const result = await api('/api/estimate', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...fields, session_id: state.session, model, color: $('color-select').value, body: $('body-select').value }) });
    state.result = result;
    renderResults(result);
    scrollToElement($('results'));
  } catch (error) {
    $('review-error').textContent = error.message;
    if (error.status === 410) {
      state.recognition = null; discardSession();
      setFormStatus('Сессия завершилась. Нажми «Распознать автомобиль» ещё раз.', 'error');
    }
  } finally { setBusy(false, 'estimate'); }
}

function dateLabel(value) {
  const parts = value.split('-');
  return parts.length === 3 ? `${parts[2]}.${parts[1]}.${parts[0]}` : value;
}

function chartDataTable(title, rows, headers) {
  return `<details class="chart-data"><summary>${escapeHTML(title)}</summary><table><thead><tr>${headers.map((v) => `<th scope="col">${escapeHTML(v)}</th>`).join('')}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((v) => `<td>${escapeHTML(v)}</td>`).join('')}</tr>`).join('')}</tbody></table></details>`;
}

function renderResults(data) {
  const vehicle = data.vehicle, valuation = data.valuation, insights = data.insights, quality = data.quality;
  const fx = valuation.rate;
  const per10k = insights.change_per_10k_rub;
  const moneyChange = (per10k > 0 ? '+' : per10k < 0 ? '−' : '') + rub(Math.abs(per10k), false);
  const comparable = data.comparables;
  const isLow = vehicle.model === data.recognition.model[0].value && data.recognition.model[0].confidence < .4;
  const rateLabel = `${new Intl.NumberFormat('ru-RU', { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(fx.gbp_rub)} ₽ · ${fx.is_fallback ? 'запасной курс' : 'ЦБ РФ'}, ${escapeHTML(fx.date)}`;
  const sourceImage = state.recognition.prepared_image;
  const historyPrice = data.charts.new_prices.length ? `<article class="chart-card"><h4>Цена новой машины</h4><p class="chart-subtitle">Историческая начальная стоимость модели в фунтах</p><div class="chart-root" id="chart-new-price"></div><p class="chart-note">Цены базовой комплектации. Без поправки на инфляцию.</p>${chartDataTable('Посмотреть данные', data.charts.new_prices.map((r) => [r.year, gbp(r.gbp)]), ['Год', 'Цена новой'])}</article>` : '';
  const historySales = data.charts.sales.length ? `<article class="chart-card"><h4>История продаж</h4><p class="chart-subtitle">Продажи новых автомобилей этой модели в Великобритании</p><div class="chart-root" id="chart-sales"></div>${chartDataTable('Посмотреть данные', data.charts.sales.map((r) => [r.year, number(r.count)]), ['Год', 'Продано'])}</article>` : '';
  $('results').innerHTML = `
    <div class="result-topline"><div><p class="eyebrow">ТВОЙ АВТОМОБИЛЬ В ЦИФРАХ</p><h2 id="results-title">Теперь картина яснее.</h2></div><div class="result-actions"><button type="button" class="button ghost" id="edit-result">Изменить данные</button><button type="button" class="button secondary" id="new-estimate">${icon('refresh')} Новая оценка</button></div></div>
    <article class="valuation-card"><div class="valuation-main"><p class="valuation-eyebrow">ОРИЕНТИРОВОЧНАЯ СТОИМОСТЬ</p><p class="valuation-amount">${escapeHTML(rub(valuation.rub))}</p><p class="valuation-range">Диапазон: <strong>${escapeHTML(rub(valuation.low_rub))} — ${escapeHTML(rub(valuation.high_rub))}</strong></p><p class="valuation-gbp">${escapeHTML(gbp(valuation.gbp))} на рынке Великобритании · ${dateLabel(valuation.date)}</p><p class="valuation-rate">1 £ = ${rateLabel}</p><p class="valuation-interval">Диапазон P10–P90 отражает разброс ошибок модели. На тестовой выборке в него попало ${metricPercent(quality.coverage)} фактических цен.</p></div>
    <div class="vehicle-summary"><img class="vehicle-summary-image" src="${sourceImage}" alt="Твой автомобиль после подготовки фото"><div><h3>${escapeHTML(vehicle.model)}</h3><p class="vehicle-facts"><span>${vehicle.year} год</span><span>${number(vehicle.mileage_km)} км</span><span>${escapeHTML(vehicle.color_label)}</span><span>${escapeHTML(vehicle.body_label)}</span></p><span class="verified-badge">${icon('check')}${vehicle.manually_corrected ? 'Характеристики исправлены вручную' : isLow ? 'Проверь распознанную модель' : 'Характеристики подтверждены'}</span></div></div></article>
    ${data.warnings.length ? `<div class="results-warnings">${data.warnings.map((warning) => `<p class="warning-item">${icon('info')}<span>${escapeHTML(warning)}</span></p>`).join('')}</div>` : ''}
    <div class="insight-grid"><article class="insight-card"><h3>При дополнительных 10 000 км</h3><p class="insight-value">${escapeHTML(moneyChange)}</p><p class="insight-note">Расчёт при остальных равных условиях</p></article><article class="insight-card"><h3>Твой средний пробег за год</h3><p class="insight-value">${number(insights.own_km_year)} км</p><p class="insight-note">Типичный для модели: ${number(insights.typical_km_year)} км</p></article><article class="insight-card"><h3>Сценарий через один год</h3><p class="insight-value">${signed(insights.one_year_change_percent, '%')}</p><p class="insight-note">При твоём темпе пробега и текущем курсе</p></article></div>
    <div class="analytics-heading"><h3>Взгляд чуть глубже.</h3><p>Выбери точку на графике, чтобы увидеть детали.</p></div>
    <div class="chart-grid"><article class="chart-card"><h4>Год выпуска и стоимость</h4><p class="chart-subtitle">Эта модель с типичным для её возраста пробегом</p><div class="chart-root" id="chart-year"></div><div class="chart-legend"><span class="legend-item"><i class="legend-dot grey"></i> При типичном пробеге</span><span class="legend-item"><i class="legend-dot"></i> Твой автомобиль</span></div><p class="chart-note">Точка твоего автомобиля учитывает указанный тобой пробег.</p>${chartDataTable('Посмотреть данные', data.charts.by_year.map((r) => [r.year, rub(r.rub)]), ['Год выпуска', 'Стоимость'])}</article>
    <article class="chart-card"><h4>Что дальше со стоимостью?</h4><p class="chart-subtitle">Сценарий на три года при ${number(insights.own_km_year)} км в год</p><div class="chart-root" id="chart-forecast"></div><p class="chart-note">Возраст и пробег меняются. Курс и уровень цен остаются текущими; изменение рынка не прогнозируется.</p>${chartDataTable('Посмотреть данные', data.charts.forecast.map((r) => [r.offset === 0 ? 'Сейчас' : `Через ${r.offset} г.`, rub(r.rub)]), ['Период', 'Стоимость'])}</article></div>
    <div class="chart-grid lower"><article class="chart-card"><h4>Цвет тоже имеет значение</h4><p class="chart-subtitle">Частые цвета этой модели и их влияние на оценку</p><div class="color-chart">${renderColors(data.charts.colors)}</div><p class="chart-note">Доля — среди объявлений этой модели. Проценты справа показывают изменение оценки относительно выбранного цвета при остальных равных условиях.</p></article>
    <article class="chart-card comparable-card"><h4>Похожие автомобили</h4><p class="chart-subtitle">Объявления с близкими возрастом и пробегом</p>${comparable ? `<p class="comparable-count">${number(comparable.count)}</p><p class="comparable-label">похожих объявлений в исходных данных</p><p class="comparable-range">${escapeHTML(rub(comparable.q25_rub))} — ${escapeHTML(rub(comparable.q75_rub))}</p><p class="comparable-caption">В этот диапазон попадает половина цен похожих объявлений.<br>Медиана: ${escapeHTML(rub(comparable.median_rub))}.</p>` : '<p class="comparable-empty">В данных недостаточно объявлений с близкими возрастом и пробегом для надёжного сравнения.</p>'}<p class="chart-note">Исторические объявления британского рынка, приведённые к уровню цен 2026 года. Это не текущие предложения о продаже.</p></article></div>
    ${historyPrice || historySales ? `<div class="historical-grid">${historyPrice}${historySales}</div>` : ''}
    ${insights.new_price_gbp ? `<p class="chart-note">В ${vehicle.year} году начальная цена новой машины (или ближайший доступный год) составляла ${escapeHTML(gbp(insights.new_price_gbp))}. Текущая оценка — ${decimalFormat.format(insights.retained_percent)}% от этой цены, без поправки на инфляцию.</p>` : ''}
    <article class="quality-card"><div class="quality-heading"><h4>Насколько точны модели?</h4><span>ТЕСТОВАЯ ВЫБОРКА</span></div><div class="quality-stats"><div><strong>${metricPercent(quality.recognition)}</strong><span>точность распознавания модели</span></div><div><strong>${metricPercent(quality.e2e_mape)}</strong><span>средняя относительная ошибка цены по фото</span></div><div><strong>${metricPercent(quality.coverage)}</strong><span>цен внутри расчётного диапазона</span></div></div><p class="quality-note">Метрики получены на тестовой части DVM-CAR. Ошибка цены при заранее известной модели — ${metricPercent(quality.price_test_mape)}. Качество на твоём фото может отличаться: ракурс, свет и автомобиль вне обучающего набора влияют на распознавание. После ручного исправления характеристики используются напрямую.</p></article>
    <p class="result-disclaimer">Оценка британского рынка в рублях. Модель не определяет техническое состояние, историю ДТП и комплектацию. Возраст рассчитывается на текущую дату, цены приведены к уровню 2026 года по индексу из исходного проекта. Это ориентир для учебного приложения.</p>`;
  $('results').hidden = false;
  $('edit-result').addEventListener('click', () => { scrollToElement($('recognition-panel')); $('model-input').focus({ preventScroll: true }); });
  $('new-estimate').addEventListener('click', () => {
    clearPhoto();
    $('vehicle-form').reset();
    ['year', 'mileage'].forEach((name) => { $(name + '-error').textContent = ''; $(name + '-input').removeAttribute('aria-invalid'); });
    setFormStatus('Фото обрабатывается на твоём компьютере.');
    scrollToElement($('estimate'));
    $('photo-input').focus({ preventScroll: true });
  });
  if (state.resizeObserver) state.resizeObserver.disconnect();
  let resizeTimer;
  state.resizeObserver = new ResizeObserver(() => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { if (state.result) renderCharts(state.result); }, 100); });
  document.querySelectorAll('.chart-root').forEach((root) => state.resizeObserver.observe(root));
  requestAnimationFrame(() => renderCharts(data));
}

const colorValues = { Black: '#25282d', White: '#fafafa', Silver: '#bdc4ce', Grey: '#777e89', Blue: '#4779bb', Red: '#b74e4e', Green: '#4a8064', Yellow: '#dfbd4d', Orange: '#d18d50', Brown: '#8f6e55', Beige: '#cbbb9a', Gold: '#ba9e54', Purple: '#8c70a1', Bronze: '#a68560', Burgundy: '#8e3d54', Pink: '#e3a4b8', Turquoise: '#68aeb0' };
function renderColors(entries) {
  const max = Math.max(...entries.map((r) => r.share), .01);
  return entries.map((entry) => `<div class="color-row${entry.selected ? ' selected' : ''}"><div class="color-name"><i class="color-swatch" style="background:${colorValues[entry.value] || '#a1a1b0'}"></i><span>${escapeHTML(entry.label)}</span></div><div class="color-bar-track"><div class="color-bar" style="width:${(entry.share / max * 100).toFixed(2)}%"></div><span class="color-share">${decimalFormat.format(entry.share * 100)}% объявлений</span></div><span class="color-effect">${signed(entry.effect_percent, '%')}</span></div>`).join('');
}

function chartFrame(width, height, title, body) {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHTML(title)}"><title>${escapeHTML(title)}</title>${body}</svg>`;
}

function renderLineChart(root, values, options = {}) {
  if (!root || !values.length) return;
  const width = Math.max(250, root.clientWidth), height = root.clientHeight || 264;
  const left = width < 340 ? 48 : 55, right = 14, top = 24, bottom = 33;
  const plotWidth = width - left - right, plotHeight = height - top - bottom;
  const all = options.selected ? values.concat([options.selected]) : values;
  const minX = Math.min(...all.map((v) => v.year)), maxX = Math.max(...all.map((v) => v.year));
  const minY = Math.max(0, Math.min(...all.map((v) => v.value)) * .72);
  const maxY = Math.max(...all.map((v) => v.value)) * 1.16 || 1;
  const x = (v) => left + ((v - minX) / (maxX - minX || 1)) * plotWidth;
  const y = (v) => top + plotHeight * (1 - (v - minY) / (maxY - minY || 1));
  const color = options.selected ? '#a3afc0' : '#0071e3';
  let content = '';
  for (let i = 0; i < 4; i++) {
    const value = minY + (maxY - minY) * i / 3, py = y(value);
    content += `<line x1="${left}" x2="${width - right}" y1="${py}" y2="${py}" stroke="#edf0f5"/><text x="${left - 10}" y="${py + 3}" text-anchor="end" font-size="10" fill="#9399a5">${escapeHTML(options.axisFormat ? options.axisFormat(value) : smallFormat.format(value))}</text>`;
  }
  const path = values.map((v, i) => (i ? 'L' : 'M') + x(v.year).toFixed(2) + ' ' + y(v.value).toFixed(2)).join(' ');
  const gradientID = root.id + '-gradient';
  content += `<defs><linearGradient id="${gradientID}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${color}" stop-opacity=".15"/><stop offset="1" stop-color="${color}" stop-opacity="0"/></linearGradient></defs>`;
  content += `<path d="${path} L${x(values[values.length - 1].year)} ${top + plotHeight} L${x(values[0].year)} ${top + plotHeight}Z" fill="url(#${gradientID})"/><path d="${path}" stroke="${color}" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round" fill="none"/>`;
  const showEvery = Math.max(1, Math.ceil(values.length / (width < 350 ? 4 : 6)));
  values.forEach((v, i) => {
    const tooltip = `${v.year}\n${options.valueFormat(v.value)}`;
    content += `<circle class="chart-point" cx="${x(v.year)}" cy="${y(v.value)}" r="3.5" fill="${color}" stroke="white" stroke-width="1.5" tabindex="0" data-tooltip="${escapeHTML(tooltip)}" aria-label="${escapeHTML(tooltip)}"/>`;
    if (i % showEvery === 0 || i === values.length - 1) content += `<text x="${x(v.year)}" y="${height - 9}" text-anchor="middle" font-size="10" fill="#9399a5">${v.year}</text>`;
  });
  if (options.selected) {
    const v = options.selected, tooltip = `Твой автомобиль · ${v.year}\n${options.valueFormat(v.value)}`;
    content += `<line x1="${x(v.year)}" x2="${x(v.year)}" y1="${y(v.value)}" y2="${top + plotHeight}" stroke="#0071e3" stroke-width="1" stroke-dasharray="3 4" opacity=".3"/><circle cx="${x(v.year)}" cy="${y(v.value)}" r="10" fill="#0071e3" opacity=".12"/><circle class="chart-point" cx="${x(v.year)}" cy="${y(v.value)}" r="5" fill="#0071e3" stroke="white" stroke-width="2" tabindex="0" data-tooltip="${escapeHTML(tooltip)}" aria-label="${escapeHTML(tooltip)}"/>`;
  }
  root.innerHTML = chartFrame(width, height, options.title, content);
}

function renderBars(root, values, options = {}) {
  if (!root || !values.length) return;
  const width = Math.max(250, root.clientWidth), height = root.clientHeight || 264;
  const left = options.compact ? 44 : 8, right = 8, top = 29, bottom = 35;
  const max = Math.max(...values.map((r) => r.value), 1) * 1.2;
  const floor = height - bottom, plotHeight = height - top - bottom, unit = (width - left - right) / values.length;
  const barWidth = Math.min(options.compact ? 33 : 62, unit * .6);
  let content = '';
  [0, .5, 1].forEach((fraction) => {
    const py = floor - plotHeight * fraction;
    content += `<line x1="${left}" x2="${width - right}" y1="${py}" y2="${py}" stroke="#edf0f5"/>`;
    if (options.compact) content += `<text x="${left - 9}" y="${py + 3}" text-anchor="end" font-size="10" fill="#9399a5">${escapeHTML(smallFormat.format(max * fraction))}</text>`;
  });
  const every = Math.max(1, Math.ceil(values.length / (width < 350 ? 4 : 6)));
  values.forEach((r, i) => {
    const x = left + unit * i + (unit - barWidth) / 2, barHeight = Math.max(2, r.value / max * plotHeight), y = floor - barHeight;
    const color = !options.compact && i === 0 ? '#0071e3' : '#b9c9dd';
    const tooltip = `${r.label}\n${options.valueFormat(r.value)}`;
    content += `<rect x="${x}" y="${y}" width="${barWidth}" height="${barHeight}" rx="5" fill="${color}" tabindex="0" data-tooltip="${escapeHTML(tooltip)}" aria-label="${escapeHTML(tooltip)}"/>`;
    if (!options.compact) content += `<text x="${x + barWidth / 2}" y="${y - 9}" text-anchor="middle" font-size="${width < 350 ? 9 : 10}" fill="${i === 0 ? '#0071e3' : '#748399'}" font-weight="500">${escapeHTML(smallFormat.format(r.value))} ₽</text>`;
    if (i % every === 0 || i === values.length - 1 || !options.compact) content += `<text x="${x + barWidth / 2}" y="${height - 10}" text-anchor="middle" font-size="10" fill="#9399a5">${escapeHTML(r.label)}</text>`;
  });
  root.innerHTML = chartFrame(width, height, options.title, content);
}

function renderCharts(data) {
  renderLineChart($('chart-year'), data.charts.by_year.map((r) => ({ year: r.year, value: r.rub })), {
    title: 'Стоимость автомобиля по году выпуска', valueFormat: rub, axisFormat: (v) => smallFormat.format(v),
    selected: { year: data.charts.selected_year.year, value: data.charts.selected_year.rub },
  });
  renderBars($('chart-forecast'), data.charts.forecast.map((r) => ({ label: r.offset ? '+' + r.offset + ' ' + (r.offset === 1 ? 'год' : 'года') : 'Сейчас', value: r.rub })), { title: 'Сценарий стоимости на три года', valueFormat: rub });
  renderLineChart($('chart-new-price'), data.charts.new_prices.map((r) => ({ year: r.year, value: r.gbp })), { title: 'Историческая цена нового автомобиля в фунтах', valueFormat: gbp, axisFormat: (v) => '£' + smallFormat.format(v) });
  renderBars($('chart-sales'), data.charts.sales.map((r) => ({ label: String(r.year), value: r.count })), { title: 'Продажи новых автомобилей в Великобритании', valueFormat: number, compact: true });
}

function showTooltip(element, event) {
  if (!element) return;
  const tooltip = $('chart-tooltip');
  tooltip.textContent = element.dataset.tooltip;
  tooltip.hidden = false;
  const rect = element.getBoundingClientRect();
  const eventX = event && Number.isFinite(event.clientX) ? event.clientX : rect.x + rect.width / 2;
  const eventY = event && Number.isFinite(event.clientY) ? event.clientY : rect.y;
  const x = Math.min(Math.max(8, eventX + 14), window.innerWidth - tooltip.offsetWidth - 8);
  const y = Math.max(8, Math.min(eventY - tooltip.offsetHeight - 12, window.innerHeight - tooltip.offsetHeight - 8));
  tooltip.style.left = x + 'px'; tooltip.style.top = y + 'px';
}

$('photo-input').addEventListener('change', (event) => { selectPhoto(event.target.files[0]); event.target.value = ''; });
$('paste-photo').addEventListener('click', pastePhotoFromClipboard);
document.addEventListener('paste', handlePhotoPaste);
$('clear-photo').addEventListener('click', (event) => { event.preventDefault(); clearPhoto(); });
$('vehicle-form').addEventListener('submit', recognize);
$('estimate-button').addEventListener('click', estimate);
['year', 'mileage'].forEach((field) => { $(field + '-input').addEventListener('input', () => { $(field + '-error').textContent = ''; $(field + '-input').removeAttribute('aria-invalid'); }); });
$('mileage-input').addEventListener('blur', () => { const raw = $('mileage-input').value.replace(/\s/g, '').replace(',', '.'); if (/^\d+(\.\d{1,2})?$/.test(raw) && Number(raw) <= 1000000) $('mileage-input').value = new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 2 }).format(Number(raw)); });
let dragDepth = 0;
$('dropzone').addEventListener('dragenter', (event) => { event.preventDefault(); if (!state.busy) { dragDepth++; $('dropzone').classList.add('dragging'); } });
$('dropzone').addEventListener('dragover', (event) => event.preventDefault());
$('dropzone').addEventListener('dragleave', (event) => { event.preventDefault(); dragDepth--; if (dragDepth <= 0) $('dropzone').classList.remove('dragging'); });
$('dropzone').addEventListener('drop', (event) => { event.preventDefault(); dragDepth = 0; $('dropzone').classList.remove('dragging'); if (event.dataTransfer.files.length > 1) { $('photo-error').textContent = 'Выбери одну фотографию с одним автомобилем.'; return; } selectPhoto(event.dataTransfer.files[0]); });
document.addEventListener('dragover', (event) => { if (event.dataTransfer && Array.from(event.dataTransfer.types).includes('Files')) event.preventDefault(); });
document.addEventListener('drop', (event) => event.preventDefault());
document.addEventListener('pointermove', (event) => { const target = event.target.closest('[data-tooltip]'); if (target) showTooltip(target, event); else $('chart-tooltip').hidden = true; });
document.addEventListener('pointerdown', (event) => { const target = event.target.closest('[data-tooltip]'); if (target) showTooltip(target, event); else $('chart-tooltip').hidden = true; });
document.addEventListener('focusin', (event) => { const target = event.target.closest('[data-tooltip]'); if (target) showTooltip(target); else $('chart-tooltip').hidden = true; });
document.addEventListener('keydown', (event) => { if (event.key === 'Escape') $('chart-tooltip').hidden = true; });
window.addEventListener('pagehide', () => { discardSession(); if (state.objectURL) URL.revokeObjectURL(state.objectURL); });

async function setupSample() {
  try {
    const response = await fetch('/static/assets/example.json', { cache: 'no-store' });
    if (!response.ok) return;
    const sample = await response.json();
    $('sample-button').hidden = false;
    $('sample-button').addEventListener('click', async () => {
      if (state.busy) return;
      $('sample-button').disabled = true;
      try {
        const imageResponse = await fetch('/static/assets/example.jpg');
        if (!imageResponse.ok) throw new Error('Не удалось загрузить пример.');
        const blob = await imageResponse.blob();
        selectPhoto(new File([blob], 'example.jpg', { type: 'image/jpeg' }));
        $('year-input').value = String(sample.year);
        $('mileage-input').value = number(sample.mileage_km);
        setFormStatus('Пример загружен. Нажми «Распознать автомобиль».');
      } catch (error) { $('photo-error').textContent = error.message; }
      finally { $('sample-button').disabled = false; }
    });
  } catch (_) { /* The example is optional; user uploads remain available. */ }
}

initialize();
setupSample();
