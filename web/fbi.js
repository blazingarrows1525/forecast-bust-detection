// Forecast Bust Intelligence: the reading travels with you.
//
// A date, a lead day and a subdivision live in the URL as
// ?date=YYYY-MM-DD&lead=N&region=ID, so a reading opened on one view is the
// reading opened on the next (docs/FRONTEND_AUDIT.md F12). Every link marked
// data-carry is rewritten to carry the current context.
//
// Values are validated before any page uses them: they end up in fetch URLs
// and on screen, and a URL is input.
"use strict";

const FBI = (() => {
  const DATE = /^\d{4}-\d{2}-\d{2}$/;
  const REGION = /^[A-Z][A-Z0-9_]{1,63}$/;
  let state = { date: null, lead: null, region: null };

  function read() {
    const q = new URLSearchParams(location.search);
    const date = q.get("date"), lead = Number(q.get("lead")), region = q.get("region");
    state = {
      date: date && DATE.test(date) ? date : null,
      lead: Number.isInteger(lead) && lead >= 1 && lead <= 10 ? lead : null,
      region: region && REGION.test(region) ? region : null,
    };
    return { ...state };
  }

  function query(s) {
    const q = new URLSearchParams();
    if (s.date) q.set("date", s.date);
    if (s.lead) q.set("lead", String(s.lead));
    if (s.region) q.set("region", s.region);
    const t = q.toString();
    return t ? "?" + t : "";
  }

  // Replace, not push: scrubbing through ten lead days should not leave ten
  // history entries behind the back button.
  function write(patch) {
    state = { ...state, ...patch };
    if (state.date && !DATE.test(state.date)) state.date = null;
    if (state.region && !REGION.test(state.region)) state.region = null;
    try { history.replaceState(null, "", location.pathname + query(state) + location.hash); }
    catch (e) { /* file:// or a sandboxed frame: the links below still carry it */ }
    relink();
  }

  function relink() {
    document.querySelectorAll("a[data-carry]").forEach((a) => {
      a.setAttribute("href", a.dataset.carry + query(state));
    });
  }

  // "Why this number?": where a reading comes from, as far as the API says.
  // Every field is read from the row or from /api/metrics' served block;
  // anything missing is said to be unavailable, never filled in.
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const na = '<span class="na">unavailable</span>';
  function provenance(row, served) {
    if (!row) return "";
    const s = served || {};
    const refused = row.status === "OUT_OF_DISTRIBUTION";
    const dist = typeof row.ood_distance === "number" ? row.ood_distance.toFixed(1) : null;
    const thr = typeof s.ood_threshold === "number" ? s.ood_threshold.toFixed(1) : null;
    const reasons = s.reasons_method === "occlusion"
      ? "occlusion: each input set to its training average, one at a time (not additive)"
      : s.reasons_method ? esc(s.reasons_method) : "TreeSHAP attributions (stores before v0.3.0)";
    const rows = [
      ["Source", "<code>/api/bulletin</code>, archive replay (not live)"],
      ["Issued / valid", `${esc(row.init_date)} / ${esc(row.valid_date)}, lead ${esc(row.lead_day)}`],
      ["Store", row.model_version ? `v${esc(row.model_version)}` + (s.generated_at ? `, built ${esc(String(s.generated_at).slice(0, 10))}` : "") : na],
      ["Base model", s.base_model ? esc(s.base_model) : na],
      ["Served number", typeof row.model_probability === "number"
        ? "the model combined with the ensemble spread (logistic combiner)" : "the model alone"],
      ["Reasons", refused ? "none: a refused row is not scored" : reasons],
      ["Refusal check", dist == null ? na
        : `distance from training ${dist}` + (thr ? ` against a refusal threshold of ${thr}` : "") +
          (refused ? ": refused" : ": within experience") +
          (s.ood_inputs ? `. Checks ${esc(s.ood_inputs)}.` : "")],
      ["Review tier", row.review_tier ? esc(row.review_tier) + (row.tier_guidance ? `: ${esc(row.tier_guidance)}` : "") : na],
      ["Data quality", row.data_quality ? `${esc(row.data_quality)}` +
        (typeof row.input_age_hours === "number" ? `, inputs ${row.input_age_hours.toFixed(0)} h old at issue` : "") : na],
    ];
    return `<details class="prov"><summary>Why this number?</summary><dl>` +
      rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("") + `</dl></details>`;
  }

  read();
  return { read, write, relink, query, provenance, get state() { return { ...state }; } };
})();

document.addEventListener("DOMContentLoaded", () => FBI.relink());
