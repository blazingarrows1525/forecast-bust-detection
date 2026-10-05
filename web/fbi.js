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

  read();
  return { read, write, relink, query, get state() { return { ...state }; } };
})();

document.addEventListener("DOMContentLoaded", () => FBI.relink());
