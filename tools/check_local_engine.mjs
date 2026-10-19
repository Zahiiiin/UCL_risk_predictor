// Compares the in-browser engine (static/local-engine.js) with the Python server.
// Run from the project folder:   node tools/check_local_engine.mjs
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const modelsPath = path.join(root, process.env.UCL_MODELS || "static/browser_models.json");

const sandbox = {
  console, atob, Math, Number, String, Object, Array, Float32Array, Int8Array, Int16Array, Int32Array, Uint8Array,
  ArrayBuffer, JSON, Error,
  fetch: async () => ({ ok: true, json: async () => JSON.parse(fs.readFileSync(modelsPath, "utf8")) }),
};
sandbox.window = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(path.join(root, "static/local-engine.js"), "utf8"), sandbox);
const engine = sandbox.window.UCL_LOCAL;

const cases = JSON.parse(fs.readFileSync(path.join(root, "tools/reference_cases.json"), "utf8"));
let bad = 0, okCount = 0, errCount = 0, maxProb = 0, maxDrv = 0;
const fail = (i, msg) => { bad++; if (bad <= 15) console.log(`case ${i}: ${msg}`); };

for (let i = 0; i < cases.length; i++) {
  const { payload, expected } = cases[i];
  const got = await engine.predict(payload);
  if (got.ok !== expected.ok) { fail(i, `ok differs: js ${got.ok}, python ${expected.ok} ${JSON.stringify(got.json)} ${JSON.stringify(expected.json)}`); continue; }
  if (!expected.ok) {
    errCount++;
    if (got.json.error !== expected.json.error) fail(i, `error text differs: "${got.json.error}" vs "${expected.json.error}"`);
    continue;
  }
  okCount++;
  const a = got.json, b = expected.json;
  if (a.predicted !== b.predicted) fail(i, `class ${a.predicted} vs ${b.predicted} ${JSON.stringify(a.probabilities)} ${JSON.stringify(b.probabilities)}`);
  for (const c of ["Low", "Medium", "High"]) {
    const d = Math.abs(a.probabilities[c] - b.probabilities[c]);
    maxProb = Math.max(maxProb, d);
    if (d > 2e-4) fail(i, `probability ${c}: ${a.probabilities[c]} vs ${b.probabilities[c]}`);
  }
  if (a.certainty.level !== b.certainty.level) fail(i, `certainty ${a.certainty.level} vs ${b.certainty.level}`);
  const bm = Object.fromEntries(b.drivers.map((d) => [d.key, d.effect_pp]));
  for (const d of a.drivers) {
    const diff = Math.abs(d.effect_pp - bm[d.key]);
    maxDrv = Math.max(maxDrv, diff);
    if (!(diff <= 0.11)) fail(i, `driver ${d.key}: ${d.effect_pp} vs ${bm[d.key]}`);
  }
  if (JSON.stringify(a.warnings) !== JSON.stringify(b.warnings)) fail(i, `warnings differ:\n  js ${JSON.stringify(a.warnings)}\n  py ${JSON.stringify(b.warnings)}`);
  if (JSON.stringify(a.model_facts) !== JSON.stringify(b.model_facts)) fail(i, "model_facts differ");
}
const opt = await engine.options();
console.log(`${cases.length} cases: ${okCount} predictions, ${errCount} rejected inputs, ${bad} problems`);
console.log(`largest probability gap ${maxProb.toExponential(2)}, largest driver gap ${maxDrv.toFixed(2)} points, options keys: ${Object.keys(opt).join(", ")}`);
process.exit(bad ? 1 : 0);
