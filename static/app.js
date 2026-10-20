/* UCL Risk Predictor front end. No libraries.
 *
 * The page answers as you type: every change to an input asks the model again.
 * The models run either inside the browser (local-engine.js) or on the server (/api/predict).
 */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const nf = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
  const MINUS = "−";
  const CLASSES = ["Low", "Medium", "High"];
  const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue("--" + name).trim();
  const themeName = () => (document.documentElement.getAttribute("data-theme") === "light" ? "light" : "dark");

  const MEANING = {
    Low: "Expected to lose under 2% of its output to unplanned events.",
    Medium: "Expected to lose 2 to 6% of its output to unplanned events.",
    High: "Expected to lose over 6% of its output to unplanned events.",
  };
  const RANGE_TEXT = { Low: "under 2% lost", Medium: "2 to 6% lost", High: "over 6% lost" };

  /* The number inputs: limits match the checks on the server. */
  const FIELDS = [
    { id: "net_capacity_mw", label: "Net capacity", noun: "net capacity", unit: "MW", min: 50, max: 2500, step: 10, driver: "size", group: "reactor", title: "Net electrical capacity: the power sent to the grid" },
    { id: "age_years", label: "Age in 2024", noun: "age", unit: "years", min: 0, max: 80, step: 1, driver: "age", group: "reactor", title: "Years since first grid connection, counted to 2024" },
    { id: "construction_years", label: "Construction time", noun: "construction time", unit: "years", min: 1, max: 60, step: 1, driver: "build", group: "reactor", title: "Years from the start of construction to first grid connection" },
    { id: "ucl_p1", label: "UCL 2010 to 2013", noun: "UCL for 2010 to 2013", unit: "%", min: 0, max: 100, step: 0.5, group: "history", title: "Average unplanned capability loss, 2010 to 2013 (IAEA PRIS)" },
    { id: "ucl_p2", label: "UCL 2014 to 2017", noun: "UCL for 2014 to 2017", unit: "%", min: 0, max: 100, step: 0.5, group: "history", title: "Average unplanned capability loss, 2014 to 2017 (IAEA PRIS)" },
  ];
  const FIELD = Object.fromEntries(FIELDS.map((f) => [f.id, f]));
  const CHOICES = ["type", "country", "nsss_group", "model_family"];

  const EXAMPLES = [
    { label: "Mid-life PWR", values: { type: "PWR", country: "USA", net_capacity_mw: 1100, age_years: 30, construction_years: 6, nsss_group: "Western_PWR", model_family: "Other", ucl_p1: 1, ucl_p2: 0.8 } },
    { label: "Ageing VVER", values: { type: "PWR", country: "RUSSIA", net_capacity_mw: 950, age_years: 33, construction_years: 7, nsss_group: "Russian_VVER", model_family: "VVER", ucl_p1: 2.5, ucl_p2: 2 } },
    { label: "Heavy-water", values: { type: "PHWR", country: "CANADA", net_capacity_mw: 650, age_years: 40, construction_years: 7, nsss_group: "Canadian_CANDU", model_family: "CANDU", ucl_p1: 4, ucl_p2: 3 } },
    { label: "Gas-cooled", values: { type: "GCR", country: "UK", net_capacity_mw: 480, age_years: 45, construction_years: 8, nsss_group: "Korean_Other", model_family: "GCR", ucl_p1: 8, ucl_p2: 9 } },
    { label: "New build", only: "1", values: { type: "PWR", country: "CHINA", net_capacity_mw: 1000, age_years: 3, construction_years: 5, nsss_group: "Chinese", model_family: "Other", ucl_p1: 1, ucl_p2: 1 } },
  ];

  const S = {
    opts: null,
    engine: null,
    model: "1",
    values: {},
    result: null,
    seq: 0,
    timer: 0,
    example: null,
    shown: null,
    liveTimer: 0,
    fadeTimer: 0,
    stageVisible: true,
  };

  /* ---------- Starting up ------------------------------------------------- */

  async function pickEngine() {
    const local = window.UCL_LOCAL;
    if (local) {
      try {
        const opts = await local.options();
        return { kind: "local", opts, predict: (p) => local.predict(p) };
      } catch (err) {
        console.warn("The in-browser models could not load, using the server instead.", err);
      }
    }
    const res = await fetch("/api/options", { headers: { Accept: "application/json" } });
    if (!res.ok) throw new Error("options failed");
    const opts = await res.json();
    return {
      kind: "server",
      opts,
      predict: async (payload) => {
        const r = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json", Accept: "application/json" },
          body: JSON.stringify(payload),
        });
        const json = await r.json().catch(() => ({}));
        return { ok: r.ok, json };
      },
    };
  }

  async function start() {
    setupTheme();
    setupInfo();
    S.atom = new Atom($("#atom"), atomCenter);
    const note = $("#server-note");
    const timer = setTimeout(() => {
      note.hidden = false;
      note.textContent = window.UCL_LOCAL
        ? "Loading the models into your browser. This takes a few seconds the first time."
        : "Starting the calculator. The free hosting plan puts the site to sleep when nobody is using it, so this can take up to a minute.";
    }, 1500);
    try {
      S.engine = await pickEngine();
      S.opts = S.engine.opts;
      clearTimeout(timer);
      note.hidden = true;
      init();
    } catch (err) {
      clearTimeout(timer);
      note.hidden = false;
      note.textContent = "The calculator could not load. Reload the page to try again.";
      console.error(err);
    }
  }

  function init() {
    buildConsole();
    buildLegend();
    buildInfo();
    S.core = new Core($("#core"));
    bindEvents();
    watchStage();
    applyExample(EXAMPLES[0], false);
    updateModelText();
    run();
  }

  /* ---------- Light and dark mode ---------------------------------------- */

  function setupTheme() {
    const btn = $("#theme-btn");
    btn.addEventListener("click", () => {
      const next = themeName() === "light" ? "dark" : "light";
      const d = document.documentElement;
      d.classList.add("theme-fade");
      clearTimeout(S.fadeTimer);
      S.fadeTimer = setTimeout(() => d.classList.remove("theme-fade"), 450);
      d.setAttribute("data-theme", next);
      try { localStorage.setItem("ucl-theme", next); } catch (e) { /* storage can be blocked */ }
    });
    // Runs for our own button and for anything else that changes the mode, such as the host page.
    new MutationObserver(onThemeChange).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    onThemeChange();
  }

  function onThemeChange() {
    const light = themeName() === "light";
    const btn = $("#theme-btn");
    btn.setAttribute("aria-label", light ? "Switch to dark mode" : "Switch to light mode");
    const meta = $('meta[name="theme-color"]');
    if (meta) meta.setAttribute("content", light ? "#e8eff8" : "#040a18");
    if (S.core) S.core.refreshTheme();
    if (S.atom) S.atom.refreshTheme();
  }

  /* ---------- The About panel -------------------------------------------- */

  function setupInfo() {
    const btn = $("#info-btn");
    const panel = $("#info");
    const close = $("#info-close");
    const setOpen = (open, focusBack) => {
      panel.hidden = !open;
      btn.setAttribute("aria-expanded", open ? "true" : "false");
      if (open) close.focus();
      else if (focusBack) btn.focus();
    };
    btn.addEventListener("click", () => setOpen(panel.hidden, true));
    close.addEventListener("click", () => setOpen(false, true));
    document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !panel.hidden) setOpen(false, true); });
    document.addEventListener("pointerdown", (e) => {
      if (!panel.hidden && !panel.contains(e.target) && !btn.contains(e.target)) setOpen(false, false);
    });
  }

  function buildInfo() {
    const m1 = S.opts.models.model1;
    const m2 = S.opts.models.model2;
    const found = (m) => `${m.confusion_matrix[1][1]} of ${m.confusion_matrix[1].reduce((a, b) => a + b, 0)}`;
    $('[data-info="medium"]').textContent =
      `In cross-validation, design only found ${found(m1)} Medium reactors and the model with past performance found ${found(m2)}.`;
  }

  /* ---------- Building the inputs ---------------------------------------- */

  const pullHtml = (key) => `
    <span class="pull is-flat" data-pull="${key}" role="img" aria-label="No effect compared with a typical reactor">
      <span class="pull__track"><span class="pull__fill" style="left:50%;width:0%"></span></span>
      <span class="pull__val" aria-hidden="true">0</span>
    </span>`;

  function controlHtml(f) {
    return `
      <div class="row__control">
        <div class="range" id="${f.id}-wrap">
          <span class="range__rail"></span><span class="range__band"></span><span class="range__fill"></span>
          <input type="range" id="${f.id}-range" min="${f.min}" max="${f.max}" step="${f.step}" aria-labelledby="${f.id}-label">
        </div>
        <label class="num" id="${f.id}-num">
          <input type="number" id="${f.id}" inputmode="decimal" step="any" min="${f.min}" max="${f.max}" aria-describedby="${f.id}-error">
          <span class="num__unit">${f.unit}</span>
        </label>
      </div>
      <p class="row__error" id="${f.id}-error" role="alert" hidden></p>`;
  }

  function sliderTile(f) {
    return `
      <div class="row" data-field="${f.id}">
        <div class="row__head">
          <label class="row__label" for="${f.id}" id="${f.id}-label" title="${esc(f.title)}">${f.label}</label>
          ${pullHtml(f.driver)}
        </div>
        ${controlHtml(f)}
      </div>`;
  }

  function sliderSub(f) {
    return `
      <div class="sub" data-field="${f.id}">
        <label class="sub__label" for="${f.id}" id="${f.id}-label" title="${esc(f.title)}">${f.label}</label>
        ${controlHtml(f)}
      </div>`;
  }

  function selectSub(id, label) {
    return `
      <div class="sub">
        <label class="sub__label" for="${id}">${label}</label>
        <select class="select" id="${id}"></select>
      </div>`;
  }

  function buildConsole() {
    const o = S.opts;
    const f = FIELD;

    $("#fields").innerHTML = `
      <div class="row row--wide" role="radiogroup" aria-labelledby="type-label">
        <div class="row__head">
          <span class="row__label" id="type-label">Reactor type</span>
          ${pullHtml("type")}
        </div>
        <div class="pills" id="type-pills"></div>
      </div>
      <div class="row" data-field="country">
        <div class="row__head">
          <label class="row__label" for="country">Country</label>
          ${pullHtml("country")}
        </div>
        <select class="select" id="country"></select>
      </div>
      ${sliderTile(f.net_capacity_mw)}
      ${sliderTile(f.age_years)}
      ${sliderTile(f.construction_years)}
      <div class="row row--wide" id="group-lineage">
        <div class="row__head">
          <span class="row__label">Maker and design</span>
          ${pullHtml("lineage")}
        </div>
        <div class="pair pair--lineage">
          ${selectSub("nsss_group", "Supplier")}
          ${selectSub("model_family", "Design family")}
        </div>
      </div>
      <div class="row row--wide is-off" id="group-history" title="Used by the model with past performance">
        <div class="row__head">
          <span class="row__label">Past performance</span>
          ${pullHtml("history")}
        </div>
        <div class="pair">
          ${sliderSub(f.ucl_p1)}
          ${sliderSub(f.ucl_p2)}
        </div>
      </div>`;

    // Type buttons
    const pills = $("#type-pills");
    const typeOrder = ["PWR", "BWR", "PHWR", "GCR", "LWGR", "FBR"];
    o.types.slice().sort((x, y) => typeOrder.indexOf(x.value) - typeOrder.indexOf(y.value)).forEach((t) => {
      const short = t.label.split(" (")[0];
      const label = document.createElement("label");
      label.className = "pill";
      label.title = t.label;
      label.innerHTML = `<input type="radio" name="type" value="${esc(t.value)}"><span>${esc(short)}</span>`;
      pills.appendChild(label);
    });

    fillSelect($("#country"), o.countries.concat([{ value: "OTHER", label: "Country not listed" }]));
    fillSelect($("#nsss_group"), o.nsss_groups);
    fillSelect($("#model_family"), o.families);

    ["model1", "model2"].forEach((key) => {
      $(`[data-proof="${key}"]`).textContent = `${Math.round(o.models[key].accuracy)}% accuracy`;
    });

    buildExamples();
  }

  function fillSelect(select, items) {
    select.innerHTML = "";
    items.forEach((it) => select.add(new Option(it.label, it.value)));
  }

  function buildExamples() {
    const wrap = $("#examples-buttons");
    wrap.innerHTML = "";
    EXAMPLES.filter((ex) => !ex.only || ex.only === S.model).forEach((ex) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "chip";
      b.textContent = ex.label;
      b.dataset.example = ex.label;
      b.addEventListener("click", () => applyExample(ex, true));
      wrap.appendChild(b);
    });
    markExample();
  }

  function markExample() {
    $$(".chip").forEach((b) => b.classList.toggle("is-on", !!S.example && b.dataset.example === S.example.label));
  }

  /* ---------- Reading and writing the inputs ------------------------------ */

  function setValue(id, value) {
    S.values[id] = String(value);
    if (id === "type") {
      const r = $(`input[name="type"][value="${value}"]`);
      if (r) r.checked = true;
    } else if (FIELD[id]) {
      $("#" + id).value = value;
      syncRange(id);
    } else {
      $("#" + id).value = value;
    }
  }

  function applyExample(ex, runNow) {
    Object.entries(ex.values).forEach(([id, v]) => setValue(id, v));
    ["nsss_group", "model_family"].forEach((id) => { $("#" + id).dataset.touched = "0"; });
    updateSelectTitles();
    S.example = ex;
    markExample();
    showExampleTag();
    clearAllErrors();
    updateRangeInfo();
    if (runNow) schedule();
  }

  function showExampleTag() {
    const tag = $("#stage-tag");
    if (S.example) {
      tag.hidden = false;
      tag.textContent = `Example: ${S.example.label}`;
    } else {
      tag.hidden = true;
    }
  }

  function userChanged() {
    if (S.example) {
      S.example = null;
      markExample();
      showExampleTag();
    }
  }

  function syncRange(id) {
    const f = FIELD[id];
    const wrap = $("#" + id + "-wrap");
    const range = $("#" + id + "-range");
    const n = Number(S.values[id]);
    const ok = Number.isFinite(n) && S.values[id] !== "";
    const v = ok ? clamp(n, f.min, f.max) : f.min;
    range.value = v;
    wrap.style.setProperty("--pct", ((v - f.min) / (f.max - f.min)).toFixed(4));
  }

  /* Hover text for the supplier list: which companies sit in the chosen group */
  function updateSelectTitles() {
    const sup = $("#nsss_group");
    const item = S.opts.nsss_groups.find((g) => g.value === sup.value);
    sup.title = item && item.hint ? `Makers include ${item.hint}` : "";
    const fam = $("#model_family");
    const f = S.opts.families.find((x) => x.value === fam.value);
    fam.title = f ? f.label : "";
  }

  /* What the model learned from: the lighter band on each slider, and an amber edge when a value is outside it */
  function updateRangeInfo() {
    const meta = S.opts.models[S.model === "2" ? "model2" : "model1"];
    FIELDS.forEach((f) => {
      const r = meta.ranges[f.id];
      const wrap = $("#" + f.id + "-wrap");
      const box = $("#" + f.id + "-num");
      if (!r) {
        wrap.style.setProperty("--lo", 0);
        wrap.style.setProperty("--hi", 0);
        box.dataset.out = "false";
        return;
      }
      const lo = clamp((r[0] - f.min) / (f.max - f.min), 0, 1);
      const hi = clamp((r[1] - f.min) / (f.max - f.min), 0, 1);
      wrap.style.setProperty("--lo", lo.toFixed(4));
      wrap.style.setProperty("--hi", hi.toFixed(4));
      const n = Number(S.values[f.id]);
      const out = Number.isFinite(n) && S.values[f.id] !== "" && (n < r[0] || n > r[1]);
      box.dataset.out = out ? "true" : "false";
      box.title = out ? "Outside the range the model learned from" : "";
      wrap.title = `Lighter band: the range the model learned from (${nf.format(r[0])} to ${nf.format(r[1])} ${f.unit})`;
    });
  }

  /* ---------- Checking the numbers --------------------------------------- */

  function fieldsInUse() {
    return FIELDS.filter((f) => f.group === "reactor" || S.model === "2");
  }

  function validateNumbers() {
    const errors = [];
    fieldsInUse().forEach((f) => {
      const raw = S.values[f.id];
      const n = Number(raw);
      let msg = "";
      if (raw === undefined || raw === "") msg = `Enter the ${f.noun}.`;
      else if (!Number.isFinite(n)) msg = `${f.label} must be a number.`;
      else if (n < f.min || n > f.max) msg = `Enter a value from ${nf.format(f.min)} to ${nf.format(f.max)} ${f.unit}.`;
      showFieldError(f.id, msg);
      if (msg) errors.push(f.id);
    });
    return errors;
  }

  function showFieldError(id, msg) {
    const err = $("#" + id + "-error");
    const box = $("#" + id + "-num");
    if (!err) return;
    err.hidden = !msg;
    err.textContent = msg;
    if (box) box.dataset.invalid = msg ? "true" : "false";
    const input = $("#" + id);
    if (msg) input.setAttribute("aria-invalid", "true"); else input.removeAttribute("aria-invalid");
  }

  function clearAllErrors() {
    FIELDS.forEach((f) => showFieldError(f.id, ""));
  }

  /* ---------- Events ------------------------------------------------------ */

  function bindEvents() {
    $$('input[name="model"]').forEach((radio) => radio.addEventListener("change", onModelChange));

    $$('input[name="type"]').forEach((radio) => radio.addEventListener("change", () => {
      userChanged();
      S.values.type = radio.value;
      suggestFromType(radio.value);
      schedule();
    }));

    ["country", "nsss_group", "model_family"].forEach((id) => {
      $("#" + id).addEventListener("change", (e) => {
        userChanged();
        S.values[id] = e.target.value;
        if (id !== "country") { e.target.dataset.touched = "1"; updateSelectTitles(); }
        schedule();
      });
    });

    FIELDS.forEach((f) => {
      const num = $("#" + f.id);
      const range = $("#" + f.id + "-range");
      range.addEventListener("input", () => {
        userChanged();
        S.values[f.id] = range.value;
        num.value = range.value;
        syncRange(f.id);
        showFieldError(f.id, "");
        updateRangeInfo();
        schedule();
      });
      num.addEventListener("input", () => {
        userChanged();
        S.values[f.id] = num.value.trim();
        syncRange(f.id);
        showFieldError(f.id, "");
        updateRangeInfo();
        schedule();
      });
    });

    window.addEventListener("beforeunload", () => { if (S.core) S.core.stop(); if (S.atom) S.atom.stop(); });
  }

  /* Choosing a reactor type fills in the supplier and design family that usually go with it. */
  function suggestFromType(type) {
    const s = S.opts.suggest_by_type[type];
    if (!s) return;
    const sup = $("#nsss_group");
    const fam = $("#model_family");
    if (sup.dataset.touched !== "1") { sup.value = s.nsss_group; S.values.nsss_group = sup.value; }
    if (fam.dataset.touched !== "1") { fam.value = s.model_family; S.values.model_family = fam.value; }
    updateSelectTitles();
  }

  function onModelChange(e) {
    S.model = e.target.value;
    if (S.example && S.example.only && S.example.only !== S.model) S.example = null;
    buildExamples();
    showExampleTag();
    updateRangeInfo();
    updateModelText();
    schedule();
  }

  function updateModelText() {
    const tile = $("#group-history");
    const on = S.model === "2";
    tile.classList.toggle("is-off", !on);
    tile.title = on ? "" : "Used by the model with past performance";
    if (on) tile.removeAttribute("inert"); else tile.setAttribute("inert", "");
  }

  /* ---------- Asking the model --------------------------------------------- */

  function schedule() {
    clearTimeout(S.timer);
    S.timer = setTimeout(run, S.engine.kind === "local" ? 20 : 260);
  }

  function payload() {
    const p = { model: S.model };
    CHOICES.forEach((id) => { p[id] = S.values[id]; });
    fieldsInUse().forEach((f) => { p[f.id] = Number(S.values[f.id]); });
    return p;
  }

  async function run() {
    const stage = $("#stage");
    const bad = validateNumbers();
    if (bad.length) {
      stage.classList.add("is-stale");
      setNotices([{ text: "Fix the highlighted input to see an updated result." }]);
      return;
    }
    const mine = ++S.seq;
    let res;
    try {
      res = await S.engine.predict(payload());
    } catch (err) {
      console.error(err);
      stage.classList.add("is-stale");
      setNotices([{ text: S.engine.kind === "local"
        ? "The calculator could not run. Reload the page and try again."
        : "Could not reach the server. If the site was idle, give it up to a minute to wake up, then change an input again." }]);
      return;
    }
    if (mine !== S.seq) return;
    stage.classList.remove("is-stale");
    if (!res.ok) {
      stage.classList.add("is-stale");
      setNotices([{ text: (res.json && res.json.error) || "Something went wrong. Check the inputs." }]);
      return;
    }
    render(res.json);
  }

  /* ---------- Showing the result ------------------------------------------ */

  function toCounts(p) {
    const raw = CLASSES.map((c) => p[c] * 100);
    const base = raw.map(Math.floor);
    let left = 100 - base.reduce((a, b) => a + b, 0);
    raw.map((v, i) => [v - base[i], i]).sort((a, b) => b[0] - a[0]).forEach(([, i]) => {
      if (left > 0) { base[i] += 1; left -= 1; }
    });
    return base;
  }

  function buildLegend() {
    $("#legend").innerHTML = CLASSES.map((c) => `
      <li class="legend__item" data-class="${c}">
        <span class="legend__pct" data-pct="${c}">0%</span>
        <span class="legend__name">${c}</span>
        <span class="legend__range">${RANGE_TEXT[c]}</span>
      </li>`).join("");
  }

  const shownNumbers = {};
  function tweenNumber(el, key, to, suffix, label) {
    const from = shownNumbers[key] === undefined ? 0 : shownNumbers[key];
    shownNumbers[key] = to;
    const finish = () => { el.textContent = label !== undefined ? label : `${to}${suffix}`; };
    if (reducedMotion() || from === to) { finish(); return; }
    const t0 = performance.now();
    const dur = 600;
    const step = (t) => {
      const u = clamp((t - t0) / dur, 0, 1);
      const e = 1 - Math.pow(1 - u, 3);
      if (u < 1 && shownNumbers[key] === to) {
        el.textContent = `${Math.round(from + (to - from) * e)}${suffix}`;
        requestAnimationFrame(step);
      } else if (shownNumbers[key] === to) {
        finish();
      }
    };
    requestAnimationFrame(step);
  }

  function render(r) {
    S.result = r;
    const cls = r.predicted;
    const probs = r.probabilities;
    const counts = toCounts(probs);
    const order = CLASSES.slice().sort((a, b) => probs[b] - probs[a]);
    const top = order[0];
    const second = order[1];
    const changed = S.shown !== null && S.shown !== cls;

    document.documentElement.dataset.class = cls;

    // The big word
    const word = $("#verdict-word");
    $("#verdict-class").dataset.class = cls;
    if (S.shown !== cls) {
      word.textContent = cls;
      if (S.shown !== null && !reducedMotion()) {
        word.classList.remove("swap");
        void word.offsetWidth;
        word.classList.add("swap");
      }
    }
    S.shown = cls;
    $("#verdict-meaning").textContent = MEANING[cls];

    // Core and legend
    S.core.setCounts(counts, !reducedMotion());
    CLASSES.forEach((c, i) => {
      const p = probs[c];
      const label = counts[i] > 0 ? undefined : (p >= 0.0005 ? "<1%" : "0%");
      tweenNumber($(`[data-pct="${c}"]`), "pct-" + c, counts[i], "%", label);
      $(`.legend__item[data-class="${c}"]`).classList.toggle("is-top", c === cls);
    });
    const summary = CLASSES.map((c, i) => `${c} ${counts[i]} percent`).join(", ");
    $("#core").setAttribute("aria-label", `Model scores: ${summary}. Predicted class: ${cls}.`);

    // One short line about confidence
    let certainty;
    if (r.certainty.level === "clear") certainty = "A clear result.";
    else if (r.certainty.level === "leaning") certainty = `Leaning ${top}, with ${second} possible.`;
    else certainty = `A close call between ${top} and ${second}.`;
    $("#certainty").textContent = certainty;

    setNotices(r.warnings || []);

    const f = r.model_facts;
    let rel = `Cross-validated on ${f.n_test} reactors: ${Math.round(f.accuracy)}% correct. Guessing from the country alone gives ${Math.round(f.baseline)}%.`;
    if (cls === "Medium" || probs.Medium >= 0.25) rel += " Medium is the hardest class.";
    $("#reliability").textContent = rel;

    updatePulls(r.drivers, cls);

    // Bottom bar on phones
    $("#dock-class").textContent = cls;
    $("#dock-scores").textContent = CLASSES.map((c, i) => `${c} ${counts[i]}%`).join("  ");

    clearTimeout(S.liveTimer);
    S.liveTimer = setTimeout(() => {
      $("#live").textContent = `Predicted class ${cls}. ${summary}.`;
    }, 700);

    if (S.atom) S.atom.setClass(CLASSES.indexOf(cls), changed);
    updateRangeInfo();
    updateDock();
  }

  function setNotices(list) {
    const seen = new Set();
    $("#notices").innerHTML = list
      .filter((w) => !seen.has(w.text) && seen.add(w.text))
      .map((w) => `<p class="notice">${esc(w.text)}</p>`).join("");
  }

  function updatePulls(drivers, cls) {
    const effect = Object.fromEntries(drivers.map((d) => [d.key, d.effect_pp]));
    $$("[data-pull]").forEach((el) => {
      const e = effect[el.dataset.pull] || 0;
      const abs = Math.abs(e);
      const frac = clamp(abs / 40, 0, 1) * 50;
      const fill = $(".pull__fill", el);
      const val = $(".pull__val", el);
      el.classList.remove("is-for", "is-against", "is-flat");
      let text;
      if (abs < 1) {
        el.classList.add("is-flat");
        fill.style.left = "50%";
        fill.style.width = "0%";
        val.textContent = "0";
        text = "Little effect compared with a typical reactor";
      } else if (e > 0) {
        el.classList.add("is-for");
        fill.style.left = "50%";
        fill.style.width = frac.toFixed(1) + "%";
        val.textContent = "+" + nf.format(abs);
        text = `Raises the chance of ${cls} by ${nf.format(abs)} points compared with a typical reactor`;
      } else {
        el.classList.add("is-against");
        fill.style.left = (50 - frac).toFixed(1) + "%";
        fill.style.width = frac.toFixed(1) + "%";
        val.textContent = MINUS + nf.format(abs);
        text = `Lowers the chance of ${cls} by ${nf.format(abs)} points compared with a typical reactor`;
      }
      el.setAttribute("aria-label", text);
      el.title = text;
    });
  }

  /* ---------- Bottom bar: only when the result is off screen --------------- */

  function updateDock() {
    $("#dock").hidden = S.stageVisible !== false || !S.result;
  }

  function watchStage() {
    if (!("IntersectionObserver" in window)) return;
    const io = new IntersectionObserver((entries) => {
      S.stageVisible = entries[0].isIntersecting;
      updateDock();
    }, { threshold: 0.05 });
    io.observe($("#stage"));
  }

  /* ---------- The reactor core -------------------------------------------- */
  /* 100 squares laid out like a core map. Each square is one point of the model's score.
     The classes fill clockwise from the top. Low squares glow calmly, High squares flicker. */

  function hexToRgb(h) {
    const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(h);
    return m ? [parseInt(m[1], 16), parseInt(m[2], 16), parseInt(m[3], 16)] : [255, 255, 255];
  }

  const classColours = () => [hexToRgb(cssVar("low") || "#44d4ff"), hexToRgb(cssVar("med") || "#ffb43a"), hexToRgb(cssVar("high") || "#ff5c85")];

  const CORE_LOOK = {
    dark: { empty: [14, 28, 64], poolIn: "rgba(40, 90, 200, 0.22)", poolOut: "rgba(10, 22, 51, 0)", edge: "rgba(160, 190, 255, 0.22)", label: "rgba(147, 168, 207, 0.85)", glow: true },
    light: { empty: [206, 220, 240], poolIn: "rgba(60, 120, 220, 0.14)", poolOut: "rgba(60, 120, 220, 0)", edge: "rgba(20, 50, 100, 0.26)", label: "rgba(51, 73, 106, 0.9)", glow: false },
  };

  class Core {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.rowWidths = [4, 6, 8, 10, 10, 12, 12, 10, 10, 8, 6, 4];
      this.n = 12;
      this.look = CORE_LOOK[themeName()];
      this.pal = classColours();
      this.motion = [
        { amp: 0.05, freq: 0.35, flick: 0 },
        { amp: 0.1, freq: 0.8, flick: 0.03 },
        { amp: 0.15, freq: 1.7, flick: 0.12 },
      ];
      this.cells = [];
      let seed = 7;
      const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
      this.rowWidths.forEach((w, gy) => {
        const start = (this.n - w) / 2;
        for (let i = 0; i < w; i++) {
          const e = this.look.empty.slice();
          this.cells.push({ gx: start + i, gy, cls: -1, rgb: e.slice(), from: e.slice(), to: e.slice(), t0: 0, dur: 1, phase: rnd() * 6.283, phase2: rnd() * 6.283 });
        }
      });
      const cx = this.n / 2, cy = this.rowWidths.length / 2;
      this.cells.forEach((c) => {
        const dx = c.gx + 0.5 - cx, dy = c.gy + 0.5 - cy;
        c.ang = Math.atan2(dx, -dy);
        if (c.ang < 0) c.ang += Math.PI * 2;
        c.rad = Math.hypot(dx, dy);
      });
      this.order = this.cells.slice().sort((a, b) => a.ang - b.ang || a.rad - b.rad);
      this.visible = true;
      this.raf = 0;
      this.frame = this.frame.bind(this);
      this.resize();
      if ("ResizeObserver" in window) new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
      if ("IntersectionObserver" in window) {
        new IntersectionObserver((e) => { this.visible = e[0].isIntersecting; if (this.visible) this.kick(); }).observe(canvas);
      }
      document.addEventListener("visibilitychange", () => { if (!document.hidden) this.kick(); });
    }

    /* Called when the page switches between light and dark */
    refreshTheme() {
      this.look = CORE_LOOK[themeName()];
      this.pal = classColours();
      const now = performance.now();
      this.cells.forEach((c) => {
        const target = c.cls >= 0 ? this.pal[c.cls] : this.look.empty;
        c.from = target.slice();
        c.to = target.slice();
        c.rgb = target.slice();
        c.t0 = now - 10000;
        c.dur = 1;
      });
      this.kick();
      if (reducedMotion()) this.draw(performance.now());
    }

    resize() {
      const box = this.canvas.parentElement.getBoundingClientRect();
      const css = Math.max(180, Math.round(box.width));
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      this.css = css;
      this.canvas.width = Math.round(css * dpr);
      this.canvas.height = Math.round(css * dpr);
      this.dpr = dpr;
      this.margin = css * 0.075;
      this.pitch = (css - this.margin * 2) / this.n;
      this.kick();
      if (reducedMotion()) this.draw(performance.now());
    }

    setCounts(counts, animate) {
      const now = performance.now();
      const edge1 = counts[0];
      const edge2 = counts[0] + counts[1];
      let moved = 0;
      this.order.forEach((cell, k) => {
        const cls = k < edge1 ? 0 : k < edge2 ? 1 : 2;
        if (cell.cls === cls) return;
        const first = cell.cls === -1;
        cell.from = cell.rgb.slice();
        cell.to = this.pal[cls].slice();
        cell.cls = cls;
        if (animate) {
          cell.t0 = now + moved * (first ? 8 : 6);
          cell.dur = first ? 700 : 520;
        } else {
          cell.rgb = cell.to.slice();
          cell.t0 = now - 10000;
          cell.dur = 1;
        }
        moved += 1;
      });
      this.kick();
    }

    stop() { if (this.raf) cancelAnimationFrame(this.raf); this.raf = 0; }

    kick() {
      if (this.raf || !this.visible || document.hidden) {
        if (!this.raf && reducedMotion()) this.draw(performance.now());
        return;
      }
      this.raf = requestAnimationFrame(this.frame);
    }

    frame(t) {
      this.raf = 0;
      if (!this.visible || document.hidden) return;
      this.draw(t);
      if (!reducedMotion()) this.raf = requestAnimationFrame(this.frame);
    }

    draw(t) {
      const ctx = this.ctx;
      const dpr = this.dpr;
      const W = this.css;
      const look = this.look;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, W);
      const still = reducedMotion();
      const sec = t / 1000;
      const m = this.margin, pitch = this.pitch;

      // Pool and its edge
      const mid = W / 2;
      const poolR = (pitch * this.n) / 2 * 1.04;
      const pool = ctx.createRadialGradient(mid, mid, poolR * 0.1, mid, mid, poolR * 1.15);
      pool.addColorStop(0, look.poolIn);
      pool.addColorStop(1, look.poolOut);
      ctx.fillStyle = pool;
      ctx.beginPath(); ctx.arc(mid, mid, poolR * 1.15, 0, 6.283); ctx.fill();
      ctx.strokeStyle = look.edge;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.arc(mid, mid, poolR, 0, 6.283); ctx.stroke();

      // Core map labels: columns 1 to 12, rows A to L
      ctx.fillStyle = look.label;
      ctx.font = `600 ${Math.max(9, W * 0.022).toFixed(1)}px "Bricolage Grotesque Variable", system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      for (let i = 0; i < this.n; i++) {
        const p = m + pitch * (i + 0.5);
        ctx.fillText(String(i + 1), p, m * 0.42);
        ctx.fillText(String.fromCharCode(65 + i), m * 0.42, p);
      }

      // Squares: glow first (dark look only), then the tiles on top
      const size = pitch * 0.8;
      const rad = size * 0.2;
      const glowR = pitch * 1.15;
      const states = this.cells.map((c) => {
        let rgb = c.to, flash = 0;
        const u = (t - c.t0) / c.dur;
        if (c.cls === -1) rgb = c.rgb;
        else if (u < 0) rgb = c.from;
        else if (u < 1 && !still) {
          const e = 1 - Math.pow(1 - u, 3);
          rgb = [c.from[0] + (c.to[0] - c.from[0]) * e, c.from[1] + (c.to[1] - c.from[1]) * e, c.from[2] + (c.to[2] - c.from[2]) * e];
          flash = Math.sin(Math.PI * u) * 0.9;
        }
        c.rgb = rgb;
        const mo = this.motion[Math.max(0, c.cls)];
        let b = 0.58;
        if (!still && c.cls >= 0) {
          b += mo.amp * Math.sin(sec * mo.freq * 6.283 + c.phase);
          b += mo.flick * (Math.sin(sec * 11 + c.phase2 * 3) + Math.sin(sec * 17.3 + c.phase)) * 0.5;
        }
        b += flash;
        const lum = clamp((rgb[0] * 0.3 + rgb[1] * 0.59 + rgb[2] * 0.11) / 255 * 1.7, 0, 1);
        return { rgb, b: clamp(b, 0.2, 1.5), lum };
      });

      if (look.glow) {
        ctx.globalCompositeOperation = "lighter";
        this.cells.forEach((c, i) => {
          const s = states[i];
          if (s.lum < 0.15) return;
          const x = m + (c.gx + 0.5) * pitch, y = m + (c.gy + 0.5) * pitch;
          const g = ctx.createRadialGradient(x, y, 0, x, y, glowR);
          const a = clamp(0.5 * s.b * s.lum, 0, 0.8);
          g.addColorStop(0, `rgba(${s.rgb[0] | 0}, ${s.rgb[1] | 0}, ${s.rgb[2] | 0}, ${a.toFixed(3)})`);
          g.addColorStop(1, `rgba(${s.rgb[0] | 0}, ${s.rgb[1] | 0}, ${s.rgb[2] | 0}, 0)`);
          ctx.fillStyle = g;
          ctx.fillRect(x - glowR, y - glowR, glowR * 2, glowR * 2);
        });
        ctx.globalCompositeOperation = "source-over";
      }

      this.cells.forEach((c, i) => {
        const s = states[i];
        const x = m + (c.gx + 0.5) * pitch - size / 2, y = m + (c.gy + 0.5) * pitch - size / 2;
        const r = `${s.rgb[0] | 0}, ${s.rgb[1] | 0}, ${s.rgb[2] | 0}`;
        roundRect(ctx, x, y, size, size, rad);
        ctx.fillStyle = `rgba(${r}, ${clamp(0.16 + 0.3 * s.b, 0, 0.8).toFixed(3)})`;
        ctx.fill();
        ctx.lineWidth = 1.2;
        ctx.strokeStyle = `rgba(${r}, ${clamp(0.4 + 0.45 * s.b, 0, 1).toFixed(3)})`;
        ctx.stroke();
        const inset = size * 0.27;
        roundRect(ctx, x + inset, y + inset, size - inset * 2, size - inset * 2, rad * 0.6);
        ctx.fillStyle = `rgba(${r}, ${clamp(0.3 + 0.6 * s.b, 0, 1).toFixed(3)})`;
        ctx.fill();
      });
    }
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  /* ---------- The atom behind the page ------------------------------------- */
  /* Three orbits circle the reactor core like electrons around a nucleus. The whole atom tilts
     toward the pointer, the electrons speed up when you move, a click sends out a ring, and a
     new predicted class sends a ring out from the core in that class colour.
     The three bright electrons use the Low, Medium and High colours. */

  function atomCenter() {
    const w = window.innerWidth, h = window.innerHeight;
    const wrap = $("#core-wrap");
    if (w >= 960 && wrap) {
      const r = wrap.getBoundingClientRect();
      if (r.width > 0) return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
    }
    return { x: w / 2, y: h * 0.42 };
  }

  class Atom {
    constructor(canvas, getCenter) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.getCenter = getCenter;
      this.aim = { x: 0, y: 0 };
      this.tilt = { x: 0.18, y: -0.12 };
      this.energy = 0;
      this.spin = 0.4;
      this.lastMove = -1e9;
      this.pointer = { x: -999, y: -999, t: 0 };
      this.ripples = [];
      this.cls = 0;
      this.raf = 0;
      this.last = 0;
      this.clock = 0;
      this.frame = this.frame.bind(this);

      // Three orbit planes, 120 degrees apart, each tilted the same amount from the view direction
      const g = (62 * Math.PI) / 180;
      this.orbits = [0, 1, 2].map((i) => {
        const phi = Math.PI / 2 + (i * 2 * Math.PI) / 3;
        const n = [Math.sin(g) * Math.cos(phi), Math.sin(g) * Math.sin(phi), Math.cos(g)];
        const l = Math.hypot(n[0], n[1]);
        const u = [n[1] / l, -n[0] / l, 0];
        const v = [n[1] * u[2] - n[2] * u[1], n[2] * u[0] - n[0] * u[2], n[0] * u[1] - n[1] * u[0]];
        return { u, v, angle: [0.7, 2.9, 4.8][i], dir: i === 1 ? -1 : 1, speed: 0.34 + i * 0.07 };
      });

      let seed = 11;
      const rnd = () => { seed = (seed * 16807) % 2147483647; return seed / 2147483647; };
      this.specks = Array.from({ length: 34 }, () => ({ x: rnd(), y: rnd(), vx: (rnd() - 0.5) * 0.012, vy: (rnd() - 0.5) * 0.012, r: 0.8 + rnd() * 1.3 }));

      this.loadTheme();
      this.resize();
      window.addEventListener("resize", () => this.resize());
      window.addEventListener("pointermove", (e) => this.onMove(e), { passive: true });
      window.addEventListener("pointerdown", (e) => this.onDown(e), { passive: true });
      document.addEventListener("visibilitychange", () => { if (!document.hidden) this.kick(); });
      this.kick();
    }

    /* Colours depend on the light or dark look */
    loadTheme() {
      this.light = themeName() === "light";
      this.pal = classColours();
      this.line = this.light ? [20, 55, 110] : [150, 190, 255];
      this.sprites = this.pal.map((rgb) => {
        const s = document.createElement("canvas");
        s.width = s.height = 64;
        const c = s.getContext("2d");
        const g = c.createRadialGradient(32, 32, 0, 32, 32, 32);
        g.addColorStop(0, `rgba(${rgb.join(",")}, 1)`);
        g.addColorStop(0.22, `rgba(${rgb.join(",")}, 0.55)`);
        g.addColorStop(1, `rgba(${rgb.join(",")}, 0)`);
        c.fillStyle = g;
        c.fillRect(0, 0, 64, 64);
        return s;
      });
    }

    refreshTheme() {
      this.loadTheme();
      this.kick();
      if (reducedMotion()) this.draw(0);
    }

    resize() {
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      this.w = window.innerWidth;
      this.h = window.innerHeight;
      this.dpr = dpr;
      this.canvas.width = Math.round(this.w * dpr);
      this.canvas.height = Math.round(this.h * dpr);
      this.radius = clamp(Math.max(this.w, this.h * 1.2) * 0.46, 280, 760);
      this.kick();
      if (reducedMotion()) this.draw(0);
    }

    onMove(e) {
      if (reducedMotion()) return;
      const now = performance.now();
      const dt = Math.max(8, now - this.pointer.t);
      const speed = Math.hypot(e.clientX - this.pointer.x, e.clientY - this.pointer.y) / dt;
      this.pointer.x = e.clientX; this.pointer.y = e.clientY; this.pointer.t = now;
      this.lastMove = now;
      this.energy = clamp(this.energy + speed * 0.05, 0, 1.4);
      this.aim.x = (e.clientX / this.w - 0.5) * 2;
      this.aim.y = (e.clientY / this.h - 0.5) * 2;
    }

    onDown(e) {
      if (reducedMotion()) return;
      this.ripples.push({ x: e.clientX, y: e.clientY, t0: this.clock, dur: 1.3, max: this.radius * 0.32, cls: this.cls });
      this.energy = clamp(this.energy + 0.5, 0, 1.4);
      this.kick();
    }

    /* A new predicted class sends a ring out from the core */
    setClass(cls, changed) {
      this.cls = cls;
      if (changed && !reducedMotion()) {
        const c = this.getCenter();
        this.ripples.push({ x: c.x, y: c.y, t0: this.clock, dur: 1.8, max: this.radius * 0.95, cls });
        this.energy = clamp(this.energy + 0.8, 0, 1.4);
      }
      this.kick();
    }

    stop() { if (this.raf) cancelAnimationFrame(this.raf); this.raf = 0; }

    kick() {
      if (this.raf || document.hidden || reducedMotion()) return;
      this.last = performance.now();
      this.raf = requestAnimationFrame(this.frame);
    }

    frame(t) {
      this.raf = 0;
      if (document.hidden) return;
      const dt = Math.min(0.05, Math.max(0.001, (t - this.last) / 1000));
      this.last = t;
      this.clock += dt;
      this.step(dt, t);
      this.draw(dt);
      this.raf = requestAnimationFrame(this.frame);
    }

    step(dt, t) {
      // Follow the pointer, or sway gently when it has been still for a while
      if (t - this.lastMove > 4000) {
        this.aim.x = Math.sin(this.clock * 0.13) * 0.55;
        this.aim.y = Math.cos(this.clock * 0.09) * 0.55;
      }
      const k = 1 - Math.exp(-dt * 2.6);
      this.tilt.x += (this.aim.y * 0.5 - this.tilt.x) * k;
      this.tilt.y += (this.aim.x * 0.5 - this.tilt.y) * k;
      this.spin += dt * 0.045;
      this.energy *= Math.exp(-dt * 0.9);
      const boost = 1 + this.energy * 2.2;
      this.orbits.forEach((o) => { o.angle += o.dir * o.speed * boost * dt; });

      // Drifting specks, pushed away from the pointer
      const near = 130;
      this.specks.forEach((s) => {
        s.x += s.vx * dt; s.y += s.vy * dt;
        if (s.x < -0.02) s.x = 1.02; else if (s.x > 1.02) s.x = -0.02;
        if (s.y < -0.02) s.y = 1.02; else if (s.y > 1.02) s.y = -0.02;
        const dx = s.x * this.w - this.pointer.x, dy = s.y * this.h - this.pointer.y;
        const d = Math.hypot(dx, dy);
        if (d < near && d > 0.1 && t - this.lastMove < 2500) {
          const f = ((1 - d / near) * 0.22 * dt) / (this.w / 1000);
          s.x += (dx / d) * f; s.y += (dy / d) * f;
        }
      });
      this.ripples = this.ripples.filter((r) => this.clock - r.t0 < r.dur);
    }

    /* Turn a point on an orbit into a screen position */
    project(o, a, rot, c, R) {
      const ca = Math.cos(a), sa = Math.sin(a);
      const x = R * (ca * o.u[0] + sa * o.v[0]);
      const y = R * (ca * o.u[1] + sa * o.v[1]);
      const z = R * (ca * o.u[2] + sa * o.v[2]);
      const x1 = x * rot.cs - y * rot.ss, y1 = x * rot.ss + y * rot.cs;
      const x2 = x1 * rot.cy + z * rot.sy, z2 = -x1 * rot.sy + z * rot.cy;
      const y3 = y1 * rot.cx - z2 * rot.sx, z3 = y1 * rot.sx + z2 * rot.cx;
      const s = (3 * R) / (3 * R - z3);
      return { x: c.x + x2 * s, y: c.y + y3 * s, d: clamp((z3 / R + 1) / 2, 0, 1), s };
    }

    draw(dt) {
      const ctx = this.ctx;
      ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      ctx.clearRect(0, 0, this.w, this.h);
      const c = this.getCenter();
      const R = this.radius;
      const rot = {
        cs: Math.cos(this.spin), ss: Math.sin(this.spin),
        cy: Math.cos(this.tilt.y), sy: Math.sin(this.tilt.y),
        cx: Math.cos(this.tilt.x), sx: Math.sin(this.tilt.x),
      };
      const [lr, lg, lb] = this.line;
      const base = this.light ? 0.2 : 0.1;
      const gain = this.light ? 0.3 : 0.34;

      // Orbit rings, brighter where they come toward you
      ctx.lineCap = "round";
      ctx.lineWidth = 1.5;
      const N = 96;
      this.orbits.forEach((o) => {
        let prev = this.project(o, 0, rot, c, R);
        for (let k = 1; k <= N; k++) {
          const cur = this.project(o, (k / N) * Math.PI * 2, rot, c, R);
          ctx.strokeStyle = `rgba(${lr}, ${lg}, ${lb}, ${(base + gain * ((prev.d + cur.d) / 2)).toFixed(3)})`;
          ctx.beginPath(); ctx.moveTo(prev.x, prev.y); ctx.lineTo(cur.x, cur.y); ctx.stroke();
          prev = cur;
        }
      });

      // Specks (dark look only; on the light look they read as dust)
      if (!this.light) {
        ctx.fillStyle = `rgba(${lr}, ${lg}, ${lb}, 0.38)`;
        this.specks.forEach((s) => {
          ctx.beginPath(); ctx.arc(s.x * this.w, s.y * this.h, s.r, 0, 6.283); ctx.fill();
        });
      }

      // Rings from clicks and from a new result
      this.ripples.forEach((r) => {
        const u = (this.clock - r.t0) / r.dur;
        const e = 1 - Math.pow(1 - u, 3);
        const rgb = this.pal[r.cls];
        ctx.strokeStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${((1 - u) * (this.light ? 0.5 : 0.55)).toFixed(3)})`;
        ctx.lineWidth = 2.2 * (1 - u) + 0.6;
        ctx.beginPath(); ctx.arc(r.x, r.y, 8 + r.max * e, 0, 6.283); ctx.stroke();
      });

      // Electrons: one bright one in a class colour, two small ones
      if (this.light) {
        // Light look: crisp ink-style marks, no glow
        ctx.lineCap = "round";
        this.orbits.forEach((o, i) => {
          const rgb = this.pal[i];
          const trail = 16;
          let prev = this.project(o, o.angle, rot, c, R);
          for (let j = 1; j <= trail; j++) {
            const cur = this.project(o, o.angle - o.dir * j * 0.04, rot, c, R);
            const k = 1 - j / (trail + 1);
            ctx.strokeStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, ${(0.6 * k).toFixed(3)})`;
            ctx.lineWidth = (1 + 4.5 * k) * prev.s;
            ctx.beginPath(); ctx.moveTo(prev.x, prev.y); ctx.lineTo(cur.x, cur.y); ctx.stroke();
            prev = cur;
          }
          const p = this.project(o, o.angle, rot, c, R);
          const r = (5.5 + 3 * p.d) * p.s;
          ctx.strokeStyle = `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0.4)`;
          ctx.lineWidth = 1.2;
          ctx.beginPath(); ctx.arc(p.x, p.y, r + 5, 0, 6.283); ctx.stroke();
          ctx.fillStyle = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
          ctx.strokeStyle = "rgba(255, 255, 255, 0.95)";
          ctx.lineWidth = 2;
          ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, 6.283); ctx.fill(); ctx.stroke();
          for (let j = 1; j <= 2; j++) {
            const q = this.project(o, o.angle + o.dir * j * 2.094, rot, c, R);
            ctx.fillStyle = `rgba(${lr}, ${lg}, ${lb}, ${(0.45 + 0.4 * q.d).toFixed(3)})`;
            ctx.beginPath(); ctx.arc(q.x, q.y, 2.4 + 1.8 * q.d, 0, 6.283); ctx.fill();
          }
        });
        return;
      }
      ctx.globalCompositeOperation = "lighter";
      this.orbits.forEach((o, i) => {
        const trail = 14;
        for (let j = trail; j >= 0; j--) {
          const p = this.project(o, o.angle - o.dir * j * 0.045, rot, c, R);
          const size = (19 + 13 * p.d) * (1 - j / (trail + 3)) * p.s;
          ctx.globalAlpha = j === 0 ? 1 : 0.5 * (1 - j / (trail + 1));
          ctx.drawImage(this.sprites[i], p.x - size, p.y - size, size * 2, size * 2);
        }
        ctx.globalAlpha = 1;
        for (let j = 1; j <= 2; j++) {
          const p = this.project(o, o.angle + o.dir * j * 2.094, rot, c, R);
          ctx.fillStyle = `rgba(${lr}, ${lg}, ${lb}, ${(0.35 + 0.45 * p.d).toFixed(3)})`;
          ctx.beginPath(); ctx.arc(p.x, p.y, 2.2 + 2 * p.d, 0, 6.283); ctx.fill();
        }
      });
      ctx.globalCompositeOperation = "source-over";
      ctx.globalAlpha = 1;
    }
  }

  start();
})();
