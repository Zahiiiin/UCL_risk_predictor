/* In-browser version of the prediction server (app.py).
 *
 * It reads artifacts/browser_models.json, which holds the same trained models,
 * and answers with the same shape of data as the /api/options and /api/predict routes.
 * The models can sit in a <script type="application/json" id="ucl-models"> tag
 * (single-file copy of the page) or be fetched from browser_models.json.
 * Must be loaded before app.js.
 */
(() => {
  "use strict";

  const MODELS_URL = (window.UCL_MODELS_URL || "browser_models.json");
  const CLASSES = ["Low", "Medium", "High"];
  let DATA = null;
  const ENGINES = {};

  /* ---------- decoding ---------------------------------------------------- */

  function typed(b64, Type) {
    const bin = atob(b64);
    const buf = new ArrayBuffer(bin.length);
    const bytes = new Uint8Array(buf);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new Type(buf);
  }

  function decodePart(p) {
    const part = {
      kind: p.kind,
      n: p.n_trees,
      offsets: typed(p.offsets, Int32Array),
      feat: typed(p.feat, Int8Array),
      thr: typed(p.thr, Float32Array),
      left: typed(p.left, Int16Array),
      right: typed(p.right, Int16Array),
    };
    if (p.kind === "forest") part.leaves = typed(p.leaves, Float32Array);
    else part.cls = typed(p.cls, Uint8Array);
    return part;
  }

  function buildEngine(bundle) {
    const parts = {};
    bundle.order.forEach((name) => { parts[name] = decodePart(bundle.parts[name]); });
    return { bundle, parts };
  }

  /* ---------- model evaluation ------------------------------------------- */

  function forestProba(part, x) {
    const out = [0, 0, 0];
    for (let t = 0; t < part.n; t++) {
      const base = part.offsets[t];
      let i = 0;
      while (part.feat[base + i] >= 0) {
        i = x[part.feat[base + i]] <= part.thr[base + i] ? part.left[base + i] : part.right[base + i];
      }
      const leaf = part.left[base + i] * 3;
      out[0] += part.leaves[leaf];
      out[1] += part.leaves[leaf + 1];
      out[2] += part.leaves[leaf + 2];
    }
    return out.map((v) => v / part.n);
  }

  function softmax(m) {
    const mx = Math.max(...m);
    const e = m.map((v) => Math.exp(v - mx));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }

  function xgbProba(part, x) {
    const margin = [0, 0, 0];
    for (let t = 0; t < part.n; t++) {
      const base = part.offsets[t];
      let i = 0;
      while (part.feat[base + i] >= 0) {
        i = x[part.feat[base + i]] < part.thr[base + i] ? part.left[base + i] : part.right[base + i];
      }
      margin[part.cls[t]] += part.thr[base + i];
    }
    return softmax(margin);
  }

  function partProba(part, x) {
    return part.kind === "forest" ? forestProba(part, x) : xgbProba(part, x);
  }

  function predictRow(engine, row) {
    const b = engine.bundle;
    const x = new Float32Array(b.features.length);
    b.features.forEach((f, j) => { x[j] = (row[f] - b.scaler.mean[j]) / b.scaler.scale[j]; });
    if (b.kind === "xgb") return xgbProba(engine.parts.xgb, x);
    const z = [];
    b.order.forEach((name) => { partProba(engine.parts[name], x).forEach((v) => z.push(v)); });
    const logits = b.meta_coef.map((coef, k) => coef.reduce((s, c, j) => s + c * z[j], b.meta_intercept[k]));
    return softmax(logits);
  }

  /* ---------- feature rows (same rules as ucl_core.design_row) ----------- */

  const ageGroup = (a) => (a <= 10 ? 0 : a <= 25 ? 1 : a <= 40 ? 2 : 3);
  const trendDir = (c) => (c > 2 ? 1 : c < -2 ? -1 : 0);

  function designRow(inp, info, temporal) {
    const age = Number(inp.age_years);
    const cap = Number(inp.net_capacity_mw);
    const op = Math.min(10, age);
    const country = String(inp.country).toUpperCase();
    const cm = info.country_mean[country];
    const row = {
      Type_enc: info.type_classes.indexOf(inp.type),
      Country_OOF: cm !== undefined ? cm : info.overall_country_mean,
      Net_Capacity_MW: cap,
      CapCat_enc: info.capcat_classes.indexOf(cap < 600 ? "<600 MWe" : ">=600 MWe"),
      Reactor_Age: age,
      Operational_Years: op,
      Age_squared: age * age,
      NSSS_Group_enc: info.nsss_classes.indexOf(inp.nsss_group),
      Construction_Duration: Number(inp.construction_years),
      Thermal_Efficiency: 33.0,
      Model_Family_enc: info.family_classes.indexOf(inp.model_family),
      Age_Group: ageGroup(age),
      Op_Category: op >= 10 ? 2 : op >= 5 ? 1 : 0,
    };
    if (temporal) {
      const p1 = Number(inp.ucf_p1);
      const p2 = Number(inp.ucf_p2);
      const ch = p2 - p1;
      Object.assign(row, {
        UCF_P1: p1,
        UCF_P2: p2,
        UCF_change_P1_P2: ch,
        UCF_mean_2period: (p1 + p2) / 2,
        UCF_trend_dir: trendDir(ch),
      });
    }
    return row;
  }

  /* ---------- input checks (same rules as app.parse_inputs) -------------- */

  class InputError extends Error {}
  const g6 = (v) => String(Number(Number(v).toPrecision(6)));

  function parseInputs(payload, key) {
    const meta = DATA.meta.models[key];
    const info = DATA[key].info;
    const warnings = [];
    const clean = {};
    const numeric = ["net_capacity_mw", "age_years", "construction_years"];
    if (key === "model2") numeric.push("ucf_p1", "ucf_p2");

    numeric.forEach((name) => {
      const raw = payload[name];
      const val = typeof raw === "number" ? raw : (typeof raw === "string" && raw.trim() !== "" ? Number(raw) : NaN);
      const label = DATA.field_names[name];
      if (!Number.isFinite(val)) throw new InputError(`${label} must be a number.`);
      const [lo, hi] = DATA.limits[name];
      if (val < lo || val > hi) throw new InputError(`${label} must be between ${g6(lo)} and ${g6(hi)}.`);
      clean[name] = val;
      const rng = meta.ranges[name];
      if (rng && (val < rng[0] || val > rng[1])) {
        const u = DATA.field_units[name];
        warnings.push({
          field: name,
          text: `${label} of ${g6(val)}${u} is outside the range the model learned from (${g6(rng[0])} to ${g6(rng[1])}${u}). Treat this result with extra caution.`,
        });
      }
    });

    const t = String(payload.type || "");
    if (!info.type_classes.includes(t)) throw new InputError("Choose a reactor type.");
    clean.type = t;

    const g = String(payload.nsss_group || "");
    if (!info.nsss_classes.includes(g)) throw new InputError("Choose a supplier family.");
    clean.nsss_group = g;

    let f = String(payload.model_family || "");
    if (!(f in DATA.family_labels)) throw new InputError("Choose a design family.");
    if (!info.family_classes.includes(f)) {
      warnings.push({
        field: "model_family",
        text: `${DATA.family_labels[f]} has no examples in the data this model learned from, so it was treated as "Other or not listed".`,
      });
      f = "Other";
    }
    clean.model_family = f;

    const c = String(payload.country || "").toUpperCase();
    if (c === "" || c === "OTHER") {
      clean.country = "OTHER";
      warnings.push({
        field: "country",
        text: "No country information was used. The country is one of the strongest signals in the model, so this result is less specific.",
      });
    } else {
      clean.country = c;
      if (!(c in info.country_mean)) {
        const nice = DATA.country_names[c] || c.charAt(0) + c.slice(1).toLowerCase();
        warnings.push({
          field: "country",
          text: `${nice} has no reactors in the data this model learned from, so an average country effect was used.`,
        });
      }
    }
    return { clean, warnings };
  }

  /* ---------- the two routes --------------------------------------------- */

  function certainty(probs) {
    const sorted = probs.slice().sort((a, b) => b - a);
    const top = sorted[0];
    const level = top >= 0.75 ? "clear" : top >= 0.5 ? "leaning" : "close";
    return { level, top, gap: top - sorted[1] };
  }

  const round = (v, d) => { const m = 10 ** d; return Math.round(v * m) / m; };

  function runPrediction(payload) {
    const key = { 1: "model1", 2: "model2", "1": "model1", "2": "model2" }[payload.model];
    if (!key) throw new InputError("Choose Model 1 or Model 2.");
    const { clean, warnings } = parseInputs(payload, key);
    const engine = ENGINES[key];
    const info = DATA[key].info;
    const temporal = key === "model2";
    const typical = Object.assign({}, DATA.meta.models[key].typical, { country: "OTHER" });
    const groups = DATA.explain_groups[key];

    const variants = [clean].concat(groups.map(([, , keys]) => {
      const swapped = Object.assign({}, clean);
      keys.forEach((k) => { swapped[k] = typical[k]; });
      return swapped;
    }));
    const probs = variants.map((v) => predictRow(engine, designRow(v, info, temporal)));
    const p = probs[0];
    const k = p.indexOf(Math.max(...p));

    const drivers = groups.map(([gid, label], i) => ({
      key: gid,
      label,
      effect_pp: round((p[k] - probs[i + 1][k]) * 100, 1),
    }));
    drivers.sort((a, b) => Math.abs(b.effect_pp) - Math.abs(a.effect_pp));

    const meta = DATA.meta.models[key];
    const cm = meta.test.confusion_matrix;
    return {
      model: key,
      predicted: CLASSES[k],
      probabilities: { Low: round(p[0], 4), Medium: round(p[1], 4), High: round(p[2], 4) },
      certainty: certainty(p),
      drivers,
      warnings,
      model_facts: {
        n_reactors: meta.n_reactors,
        n_test: meta.test.n_test,
        accuracy: meta.test.accuracy,
        medium_found: cm[1][1],
        medium_total: cm[1].reduce((a, b) => a + b, 0),
      },
    };
  }

  /* ---------- public object used by app.js ------------------------------- */

  async function load() {
    if (DATA) return;
    const inline = typeof document !== "undefined" ? document.getElementById("ucl-models") : null;
    if (inline) {
      DATA = JSON.parse(inline.textContent);
    } else {
      const res = await fetch(MODELS_URL);
      if (!res.ok) throw new Error("Could not load the model file.");
      DATA = await res.json();
    }
    ENGINES.model1 = buildEngine(DATA.model1);
    ENGINES.model2 = buildEngine(DATA.model2);
  }

  window.UCL_LOCAL = {
    slowNote: "Loading the models into your browser. This takes a few seconds the first time.",
    async options() {
      await load();
      return DATA.options;
    },
    async predict(payload) {
      await load();
      try {
        return { ok: true, json: runPrediction(payload) };
      } catch (err) {
        if (err instanceof InputError) return { ok: false, json: { error: err.message } };
        throw err;
      }
    },
    _internals: { get data() { return DATA; }, get engines() { return ENGINES; }, runPrediction, predictRow, designRow, partProba },
  };
})();
