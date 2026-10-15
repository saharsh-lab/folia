/* Folia — frontend logic. Every number shown comes from the Flask API:
   /predict (live model output), /model-info (metrics.json), /classes, /samples. */
(() => {
  'use strict';

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const ALLOWED = ['.jpg', '.jpeg', '.png', '.webp', '.bmp'];
  const MAX_BYTES = 10 * 1024 * 1024;

  /* ── Label formatting ──────────────────────────────────────────────── */
  const PLANT_NAMES = {
    'Corn_(maize)': 'Corn',
    'Cherry_(including_sour)': 'Cherry',
    'Pepper,_bell': 'Bell pepper',
  };
  const CONDITION_NAMES = {
    'Cercospora_leaf_spot Gray_leaf_spot': 'Gray leaf spot',
    'Common_rust_': 'Common rust',
    'Esca_(Black_Measles)': 'Esca (black measles)',
    'Leaf_blight_(Isariopsis_Leaf_Spot)': 'Isariopsis leaf spot',
    'Haunglongbing_(Citrus_greening)': 'Huanglongbing (citrus greening)',
    'Spider_mites Two-spotted_spider_mite': 'Two-spotted spider mites',
    'Tomato_Yellow_Leaf_Curl_Virus': 'Yellow leaf curl virus',
    'Tomato_mosaic_virus': 'Mosaic virus',
    'healthy': 'Healthy',
  };
  const sentence = s => {
    const t = s.replace(/_/g, ' ').replace(/\s+/g, ' ').trim();
    return t.charAt(0).toUpperCase() + t.slice(1).toLowerCase();
  };
  function parseClass(raw) {
    const [p, c = ''] = raw.split('___');
    return {
      raw,
      plant: PLANT_NAMES[p] || sentence(p),
      condition: CONDITION_NAMES[c] || sentence(c),
      healthy: c === 'healthy',
    };
  }
  const pct = (v, digits = 1) => `${(v * 100).toFixed(digits)}%`;
  const fmtInt = n => n.toLocaleString('en-US');

  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') node.className = v;
      else if (k === 'text') node.textContent = v;
      else if (k === 'style') Object.assign(node.style, v);
      else if (k.startsWith('data-')) node.setAttribute(k, v);
      else node.setAttribute(k, v);
    }
    for (const c of children) if (c != null) node.append(c);
    return node;
  }

  async function getJSON(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url} → ${res.status}`);
    return res.json();
  }

  /* ── Theme ─────────────────────────────────────────────────────────── */
  const themeBtn = $('#theme-toggle');
  const systemDark = window.matchMedia('(prefers-color-scheme: dark)');
  const currentTheme = () => document.documentElement.dataset.theme || (systemDark.matches ? 'dark' : 'light');
  function syncThemeLabel() {
    themeBtn.setAttribute('aria-label', currentTheme() === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
  }
  themeBtn.addEventListener('click', () => {
    const next = currentTheme() === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem('folia-theme', next); } catch (e) { /* storage unavailable */ }
    syncThemeLabel();
  });
  syncThemeLabel();

  /* ── Nav: scrolled state + current section ─────────────────────────── */
  const nav = $('#nav');
  const onScroll = () => nav.classList.toggle('scrolled', window.scrollY > 8);
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();

  const navLinks = $$('.nav-links a');
  const sectionObserver = new IntersectionObserver(entries => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      navLinks.forEach(a => a.setAttribute('aria-current', a.hash === `#${e.target.id}` ? 'true' : 'false'));
    }
  }, { rootMargin: '-45% 0px -50% 0px' });
  ['diagnose', 'model', 'crops', 'method'].forEach(id => { const s = document.getElementById(id); if (s) sectionObserver.observe(s); });
  sectionObserver.observe($('.hero'));

  /* ── Reveal on scroll ──────────────────────────────────────────────── */
  const revealObserver = new IntersectionObserver(entries => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      e.target.classList.add('in');
      revealObserver.unobserve(e.target);
    }
  }, { rootMargin: '0px 0px -8% 0px', threshold: 0.05 });
  function observeReveals(root = document) {
    $$('.reveal:not(.in)', root).forEach((node, i) => {
      if (!reduceMotion) node.style.transitionDelay = `${Math.min(i % 4, 3) * 70}ms`;
      revealObserver.observe(node);
    });
  }
  observeReveals();

  /* ── 3D stage: pointer tilt ────────────────────────────────────────── */
  const stage3d = $('#stage-3d');
  const hero = $('.hero');
  if (stage3d && !reduceMotion && window.matchMedia('(pointer: fine)').matches) {
    let raf = 0;
    hero.addEventListener('pointermove', e => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const r = hero.getBoundingClientRect();
        const x = (e.clientX - r.left) / r.width - 0.5;
        const y = (e.clientY - r.top) / r.height - 0.5;
        stage3d.style.setProperty('--ry', `${-14 + x * 22}deg`);
        stage3d.style.setProperty('--rx', `${6 - y * 14}deg`);
      });
    });
    hero.addEventListener('pointerleave', () => {
      stage3d.style.removeProperty('--ry');
      stage3d.style.removeProperty('--rx');
    });
  }

  /* ── Tooltip (enhancement only — every value is also printed) ──────── */
  const tip = $('#tooltip');
  function showTip(target, html, evt) {
    tip.replaceChildren(...html);
    tip.classList.add('show');
    positionTip(target, evt);
  }
  function positionTip(target, evt) {
    const pad = 12;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let x, y;
    if (evt && evt.clientX) { x = evt.clientX + 14; y = evt.clientY - h - 10; }
    else { const r = target.getBoundingClientRect(); x = r.left + r.width / 2 - w / 2; y = r.top - h - 8; }
    x = Math.max(pad, Math.min(x, window.innerWidth - w - pad));
    if (y < pad) y = (evt && evt.clientY ? evt.clientY : target.getBoundingClientRect().bottom) + 16;
    tip.style.left = `${x}px`;
    tip.style.top = `${y}px`;
  }
  const hideTip = () => tip.classList.remove('show');
  function bindTip(node, contentFn) {
    node.addEventListener('pointerenter', e => showTip(node, contentFn(), e));
    node.addEventListener('pointermove', e => positionTip(node, e));
    node.addEventListener('pointerleave', hideTip);
    node.addEventListener('focus', () => showTip(node, contentFn()));
    node.addEventListener('blur', hideTip);
  }
  window.addEventListener('scroll', hideTip, { passive: true });

  /* ── Bar list chart ────────────────────────────────────────────────── */
  // rows: [{ label: Node|string, value: 0..1, muted?: bool, tip: () => Node[] }]
  function renderBars(listEl, rows, { focusable = false } = {}) {
    const items = rows.map(r => {
      const bar = el('div', { class: `bar${r.muted ? ' is-muted' : ''}` });
      const li = el('li', { class: 'bar-row' },
        el('span', { class: 'bar-label' }, r.label),
        el('div', { class: 'bar-track', 'aria-hidden': 'true' }, bar),
        el('span', { class: 'bar-val', text: pct(r.value) }),
      );
      if (focusable) li.tabIndex = 0;
      if (r.tip) bindTip(li, r.tip);
      li._bar = bar; li._value = r.value;
      return li;
    });
    listEl.replaceChildren(...items);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      items.forEach(li => { li._bar.style.width = `${Math.max(li._value * 100, 0.5)}%`; });
    }));
  }
  const tipLines = (title, ...lines) => [el('strong', { text: title }), ...lines.map(l => el('div', { text: l }))];
  const classLabel = c => {
    const frag = document.createDocumentFragment();
    frag.append(`${c.plant} `, el('small', { text: `· ${c.condition}` }));
    return frag;
  };

  /* ── Model info & charts ───────────────────────────────────────────── */
  let perClass = [];
  let sortMode = 'asc';

  function setStat(name, value) {
    $$(`[data-stat="${name}"]`).forEach(n => { n.textContent = value; });
  }

  async function loadModelInfo() {
    let info;
    try { info = await getJSON('/model-info'); } catch (e) { info = null; }
    const m = info && info.metrics;
    if (!m) {
      $('#model-unavailable').hidden = false;
      $('#model-grid').hidden = true;
      $('#perclass-card').hidden = true;
      return;
    }

    const sizes = m.split_sizes;
    const total = sizes.train + sizes.val + sizes.test;
    setStat('accuracy', pct(m.test_accuracy));
    setStat('images', fmtInt(total));
    $('#test-note').textContent = `on ${fmtInt(sizes.test)} held-out images`;

    // Baseline vs Folia
    const base = info.baseline;
    renderBars($('#cmp-chart'), [
      { label: 'Original CNN', value: base.val_accuracy, muted: true,
        tip: () => tipLines('Original CNN', base.name, `${pct(base.val_accuracy)} validation accuracy`) },
      { label: 'Folia', value: m.test_accuracy,
        tip: () => tipLines('Folia', info.architecture, `${pct(m.test_accuracy)} test accuracy`) },
    ], { focusable: true });

    // Dataset split (ordinal: train → val → test)
    const parts = [
      { key: 'train', name: 'Train', n: sizes.train, color: 'var(--split-1)' },
      { key: 'val', name: 'Validation', n: sizes.val, color: 'var(--split-2)' },
      { key: 'test', name: 'Test', n: sizes.test, color: 'var(--split-3)' },
    ];
    const stack = $('#split-chart');
    stack.setAttribute('aria-label', parts.map(p => `${p.name} ${fmtInt(p.n)} images (${pct(p.n / total, 0)})`).join(', '));
    const segs = parts.map(p => {
      const seg = el('div', { class: 'stack-seg', style: { background: p.color } });
      bindTip(seg, () => tipLines(p.name, `${fmtInt(p.n)} images · ${pct(p.n / total, 0)}`));
      return seg;
    });
    stack.replaceChildren(...segs);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      segs.forEach((s, i) => { s.style.width = `${(parts[i].n / total) * 100}%`; });
    }));
    $('#split-legend').replaceChildren(...parts.map(p => el('li', {},
      el('span', { class: 'sw', style: { background: p.color }, 'aria-hidden': 'true' }),
      el('span', { text: p.name }),
      el('span', { class: 'lv' }, fmtInt(p.n), el('span', { class: 'lp', text: pct(p.n / total, 0) })),
    )));

    // Per-class accuracy
    perClass = Object.entries(m.per_class_accuracy).map(([raw, acc]) => ({ ...parseClass(raw), acc }));
    const crops = [...new Set(perClass.map(c => c.plant))].sort();
    $('#crop-filter').append(...crops.map(c => el('option', { value: c, text: c })));
    renderPerClass();
  }

  function renderPerClass() {
    const crop = $('#crop-filter').value;
    let rows = perClass.filter(c => !crop || c.plant === crop);
    rows = sortMode === 'asc'
      ? rows.sort((a, b) => a.acc - b.acc)
      : rows.sort((a, b) => a.plant.localeCompare(b.plant) || a.condition.localeCompare(b.condition));

    renderBars($('#perclass-chart'), rows.map(c => ({
      label: classLabel(c),
      value: c.acc,
      tip: () => tipLines(`${c.plant} · ${c.condition}`, `${pct(c.acc)} of test images correct`),
    })));

    $('#perclass-table').replaceChildren(...rows.map(c => el('tr', {},
      el('td', { text: c.plant }), el('td', { text: c.condition }), el('td', { class: 'num', text: pct(c.acc) }),
    )));
  }

  $('#crop-filter').addEventListener('change', renderPerClass);
  $$('.seg button').forEach(btn => btn.addEventListener('click', () => {
    sortMode = btn.dataset.sort;
    $$('.seg button').forEach(b => b.setAttribute('aria-checked', String(b === btn)));
    renderPerClass();
  }));

  /* ── Crops grid ────────────────────────────────────────────────────── */
  const LEAF_ICON = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 19c-.5-7.5 4.5-13.5 14-14 .4 9-5.4 14.2-14 14Zm1.2-1.6c3.4-3.9 6.4-6.9 10-9.6-4 1.9-7.2 4.9-10 9.6Z"/></svg>';

  async function loadCrops() {
    let classes;
    try { ({ classes } = await getJSON('/classes')); } catch (e) { return; }
    const byCrop = new Map();
    classes.map(parseClass).forEach(c => {
      if (!byCrop.has(c.plant)) byCrop.set(c.plant, []);
      byCrop.get(c.plant).push(c);
    });
    const crops = [...byCrop.entries()].sort((a, b) => a[0].localeCompare(b[0]));
    setStat('crops', String(crops.length));
    setStat('classes', String(classes.length));

    $('#crop-grid').replaceChildren(...crops.map(([plant, list]) => {
      list.sort((a, b) => (a.healthy - b.healthy) || a.condition.localeCompare(b.condition));
      const icon = el('span', { class: 'crop-icon' });
      icon.innerHTML = LEAF_ICON;
      const diseases = list.filter(c => !c.healthy).length;
      const hasHealthy = list.some(c => c.healthy);
      const countText = !diseases ? 'Healthy only'
        : `${diseases} disease${diseases > 1 ? 's' : ''}${hasHealthy ? ' + healthy' : ''}`;
      return el('li', { class: 'crop reveal' },
        el('div', { class: 'crop-head' }, icon,
          el('div', {}, el('h3', { text: plant }),
            el('span', { class: 'crop-count', text: countText }))),
        el('ul', {}, ...list.map(c => el('li', { class: c.healthy ? 'is-healthy' : '', text: c.condition }))),
      );
    }));
    observeReveals($('#crop-grid'));
  }

  /* ── Upload & predict ──────────────────────────────────────────────── */
  const dropZone = $('#drop-zone');
  const fileInput = $('#file-input');
  const preview = $('#preview');
  const predictBtn = $('#predict-btn');
  const clearBtn = $('#clear-btn');
  const btnLabel = $('.btn-label', predictBtn);
  const resultCard = $('.result-card');
  let selectedFile = null;
  let previewUrl = null;
  let busy = false;

  const views = { empty: $('#result-empty'), error: $('#result-error'), result: $('#result') };
  function showView(name) {
    for (const [k, v] of Object.entries(views)) v.hidden = k !== name;
    $('#result-meta').textContent = name === 'result' ? 'Live model output' : name === 'error' ? 'Something went wrong' : 'Awaiting specimen';
  }

  function showError(msg) {
    $('#error-text').textContent = msg;
    showView('error');
  }

  function setFile(file) {
    const ext = (file.name.match(/\.[^.]+$/) || [''])[0].toLowerCase();
    if (!ALLOWED.includes(ext)) { showError(`Unsupported file type "${ext || file.type}". Use JPG, PNG, WEBP or BMP.`); return false; }
    if (file.size > MAX_BYTES) { showError('That image is larger than 10 MB. Please choose a smaller one.'); return false; }

    selectedFile = file;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    preview.src = previewUrl;
    preview.alt = `Selected leaf: ${file.name}`;
    dropZone.classList.add('has-image');
    $('#file-chip').hidden = false;
    $('#file-name').textContent = file.name;
    dropZone.setAttribute('aria-label', `Selected ${file.name}. Press Enter to choose a different image.`);
    predictBtn.disabled = false;
    clearBtn.hidden = false;
    showView('empty');
    return true;
  }

  function clearFile() {
    selectedFile = null;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    preview.removeAttribute('src');
    preview.alt = '';
    dropZone.classList.remove('has-image');
    $('#file-chip').hidden = true;
    dropZone.setAttribute('aria-label', 'Choose a leaf image to upload');
    fileInput.value = '';
    predictBtn.disabled = true;
    clearBtn.hidden = true;
    showView('empty');
    dropZone.focus();
  }

  dropZone.addEventListener('click', () => !busy && fileInput.click());
  dropZone.addEventListener('keydown', e => {
    if ((e.key === 'Enter' || e.key === ' ') && !busy) { e.preventDefault(); fileInput.click(); }
  });
  ['dragenter', 'dragover'].forEach(t => dropZone.addEventListener(t, e => { e.preventDefault(); dropZone.classList.add('drag-over'); }));
  ['dragleave', 'drop'].forEach(t => dropZone.addEventListener(t, () => dropZone.classList.remove('drag-over')));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    const f = e.dataTransfer.files[0];
    if (f && !busy) setFile(f);
  });
  fileInput.addEventListener('change', () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });
  clearBtn.addEventListener('click', clearFile);
  predictBtn.addEventListener('click', predict);

  function setBusy(on) {
    busy = on;
    predictBtn.disabled = on || !selectedFile;
    predictBtn.classList.toggle('is-loading', on);
    btnLabel.textContent = on ? 'Analysing…' : 'Analyse leaf';
    dropZone.classList.toggle('analysing', on);
    clearBtn.disabled = on;
    $$('.sample').forEach(b => { b.disabled = on; });
    views.result.classList.toggle('is-stale', on && !views.result.hidden);
  }

  async function predict() {
    if (!selectedFile || busy) return;
    setBusy(true);
    $('#result-live').textContent = 'Analysing leaf…';
    const body = new FormData();
    body.append('image', selectedFile);
    const started = performance.now();
    try {
      const res = await fetch('/predict', { method: 'POST', body });
      let data = null;
      try { data = await res.json(); } catch (e) { /* non-JSON error */ }
      if (!res.ok) throw new Error((data && data.error) || `Server returned ${res.status}`);
      // Let the scan animation register before swapping content.
      const wait = reduceMotion ? 0 : Math.max(0, 650 - (performance.now() - started));
      await new Promise(r => setTimeout(r, wait));
      renderResult(data);
    } catch (err) {
      const msg = err instanceof TypeError ? 'Could not reach the server. Is the Flask app running?' : err.message;
      showError(msg);
      $('#result-live').textContent = `Error: ${msg}`;
    } finally {
      setBusy(false);
    }
  }

  /* ── Render result ─────────────────────────────────────────────────── */
  const ICONS = {
    none: '<svg viewBox="0 0 24 24"><path d="m5 12.5 4.5 4.5L19 7.5"/></svg>',
    moderate: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7.5v5M12 16.2v.01"/></svg>',
    high: '<svg viewBox="0 0 24 24"><path d="M12 3 2 20h20L12 3Z"/><path d="M12 10v4M12 17v.01"/></svg>',
    critical: '<svg viewBox="0 0 24 24"><path d="M8 3h8l5 5v8l-5 5H8l-5-5V8z"/><path d="M12 8v5M12 16.2v.01"/></svg>',
  };
  const SEVERITY_LABEL = { none: 'Healthy', moderate: 'Moderate severity', high: 'High severity', critical: 'Critical' };

  function splitSteps(text) {
    const parts = text.split(/\.\s+(?=[A-Z])/).map(s => s.trim()).filter(Boolean);
    return parts.map((s, i) => (i < parts.length - 1 ? `${s}.` : s));
  }

  function renderResult(d) {
    const c = parseClass(d.predicted_class);
    const sev = SEVERITY_LABEL[d.severity] ? d.severity : 'moderate';

    const pill = $('#status-pill');
    pill.className = `status-pill status-${sev}`;
    pill.innerHTML = ICONS[sev];
    pill.append(SEVERITY_LABEL[sev]);

    $('#result-plant').textContent = c.plant;
    $('#result-disease').textContent = c.healthy ? 'Healthy leaf' : c.condition;
    const cause = $('#result-cause');
    cause.textContent = d.cause || '';
    cause.hidden = !d.cause;

    // Confidence gauge
    const conf = Math.max(0, Math.min(1, d.confidence || 0));
    const C = 2 * Math.PI * 50;
    const fill = $('#gauge-fill');
    fill.style.transition = 'none';
    fill.style.strokeDashoffset = String(C);
    void fill.getBoundingClientRect();
    fill.style.transition = '';
    requestAnimationFrame(() => { fill.style.strokeDashoffset = String(C * (1 - conf)); });
    animateNumber($('#gauge-num'), Math.round(conf * 100));
    const level = conf >= 0.9 ? 'High confidence' : conf >= 0.6 ? 'Moderate confidence' : 'Low confidence';
    $('#gauge-label').textContent = level;
    $('#low-conf').hidden = conf >= 0.6;

    $('#result-desc').textContent = d.description || '—';
    $('#result-treatment').replaceChildren(...splitSteps(d.treatment || '—').map(s => el('li', { text: s })));

    const top = d.top5 || [{ class: d.predicted_class, confidence: d.confidence }];
    renderBars($('#top5-list'), top.map((t, i) => {
      const tc = parseClass(t.class);
      return {
        label: classLabel(tc),
        value: t.confidence,
        muted: i > 0,
        tip: () => tipLines(`${tc.plant} · ${tc.condition}`, `${pct(t.confidence, 2)} probability${i === 0 ? ' — top prediction' : ''}`),
      };
    }), { focusable: true });

    showView('result');
    views.result.classList.remove('is-stale');
    // restart entrance animation
    views.result.style.animation = 'none'; void views.result.offsetWidth; views.result.style.animation = '';

    $('#result-live').textContent =
      `Diagnosis: ${c.plant}, ${c.healthy ? 'healthy' : c.condition}. ${Math.round(conf * 100)}% confidence. ${SEVERITY_LABEL[sev]}.`;

    const r = resultCard.getBoundingClientRect();
    if (r.top > window.innerHeight * 0.6 || r.bottom < 0) {
      resultCard.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'start' });
    }
  }

  function animateNumber(node, to) {
    if (reduceMotion) { node.textContent = String(to); return; }
    const dur = 1100, t0 = performance.now();
    const step = now => {
      const p = Math.min(1, (now - t0) / dur);
      node.textContent = String(Math.round(to * (1 - Math.pow(1 - p, 3))));
      if (p < 1) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  /* ── Sample images (real files, real predictions) ──────────────────── */
  async function loadSamples() {
    let samples;
    try { ({ samples } = await getJSON('/samples')); } catch (e) { return; }
    if (!samples.length) return;
    const row = $('.samples-row');
    row.replaceChildren(...samples.map(name => {
      const label = sentence(name.replace(/\.[^.]+$/, '').replace(/^test_/, ''));
      const url = `/samples/${encodeURIComponent(name)}`;
      const btn = el('button', { class: 'sample', type: 'button', 'aria-label': `Analyse sample: ${label}` },
        el('img', { src: url, alt: '', loading: 'lazy', width: '34', height: '34' }),
        el('span', { text: label }));
      btn.addEventListener('click', async () => {
        if (busy) return;
        try {
          const blob = await (await fetch(url)).blob();
          if (setFile(new File([blob], name, { type: blob.type || 'image/jpeg' }))) predict();
        } catch (e) { showError('Could not load the sample image.'); }
      });
      return btn;
    }));
    $('#samples').hidden = false;
  }

  /* ── Boot ──────────────────────────────────────────────────────────── */
  loadModelInfo();
  loadCrops();
  loadSamples();
})();
