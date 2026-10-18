// Forecast Bust Intelligence: the reading travels with you.
//
// A date, a lead day and a subdivision live in the URL as
// ?date=YYYY-MM-DD&lead=N&region=ID, so a reading opened on one view is the
// reading opened on the next (docs/FRONTEND_AUDIT.md F12). Every link marked
// data-carry is rewritten to carry the current context.
//
// Open views also stay in step while they are open side by side: a reading
// written on one is broadcast to the others (BroadcastChannel, same origin,
// never the network), so selecting a subdivision on the map selects the same
// column in the 3-D view and the reverse. One canonical state, many views.
//
// Values are validated before any page uses them: they end up in fetch URLs
// and on screen, and a URL -- or a message from another tab -- is input.
"use strict";

const FBI = (() => {
  const DATE = /^\d{4}-\d{2}-\d{2}$/;
  const REGION = /^[A-Z][A-Z0-9_]{1,63}$/;
  let state = { date: null, lead: null, region: null };

  function clean(s) {
    const lead = Number(s.lead);
    return {
      date: s.date && DATE.test(s.date) ? s.date : null,
      lead: Number.isInteger(lead) && lead >= 1 && lead <= 10 ? lead : null,
      region: s.region && REGION.test(s.region) ? s.region : null,
    };
  }

  function read() {
    const q = new URLSearchParams(location.search);
    state = clean({ date: q.get("date"), lead: q.get("lead"), region: q.get("region") });
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

  // ---------------------------------------------------------- live sync
  const channel = ("BroadcastChannel" in window) ? new BroadcastChannel("fbi-reading") : null;
  const listeners = [];
  let applying = false;
  if (channel) {
    channel.onmessage = (ev) => {
      const d = ev.data || {};
      if (d.kind !== "reading" || !d.state) return;
      const remote = clean(d.state);
      // A remote reading is applied without being re-broadcast, so two views
      // cannot echo one change back and forth.
      applying = true;
      try { listeners.forEach((fn) => fn(remote, d.view)); }
      finally { applying = false; }
    };
  }
  function onRemote(fn) { listeners.push(fn); }

  // Replace, not push: scrubbing through ten lead days should not leave ten
  // history entries behind the back button.
  function write(patch) {
    const before = query(state);
    state = { ...state, ...patch };
    if (state.date && !DATE.test(state.date)) state.date = null;
    if (state.region && !REGION.test(state.region)) state.region = null;
    try { history.replaceState(null, "", location.pathname + query(state) + location.hash); }
    catch (e) { /* file:// or a sandboxed frame: the links below still carry it */ }
    relink();
    if (channel && !applying && query(state) !== before) {
      try { channel.postMessage({ kind: "reading", view: location.pathname, state: { ...state } }); }
      catch (e) { /* a closed channel is not an error worth surfacing */ }
    }
  }

  function relink() {
    document.querySelectorAll("a[data-carry]").forEach((a) => {
      a.setAttribute("href", a.dataset.carry + query(state));
    });
  }

  // ------------------------------------------------------------ helpers
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const pct = (x) => (typeof x === "number" ? (100 * x).toFixed(1) + "%" : null);
  const code = (L) => "D" + String(L).padStart(2, "0");
  const REDUCED = matchMedia("(prefers-reduced-motion: reduce)");

  // ----------------------------------------------------------- tooltips
  // One shared plate. A `title` on a control is lifted into data-tip so the
  // browser's own tooltip (slow, unstyled, never on keyboard focus) does not
  // double it; the text stays in the accessible description.
  let tipEl = null, tipTimer = null, tipFor = null;
  function placeTip(target) {
    const r = target.getBoundingClientRect(), t = tipEl.getBoundingClientRect();
    let x = r.left + r.width / 2 - t.width / 2;
    x = Math.max(8, Math.min(window.innerWidth - t.width - 8, x));
    let y = r.bottom + 8;
    if (y + t.height > window.innerHeight - 8) y = r.top - t.height - 8;
    tipEl.style.left = Math.round(x) + "px";
    tipEl.style.top = Math.round(Math.max(8, y)) + "px";
  }
  function showTip(target) {
    const text = target.getAttribute("data-tip");
    if (!text) return;
    if (!tipEl) {
      tipEl = document.createElement("div");
      tipEl.className = "fbi-tip"; tipEl.id = "fbi-tip"; tipEl.setAttribute("role", "tooltip");
      document.body.appendChild(tipEl);
    }
    tipEl.textContent = text;
    tipFor = target;
    target.setAttribute("aria-describedby", "fbi-tip");
    placeTip(target);
    tipEl.classList.add("on");
  }
  function hideTip() {
    clearTimeout(tipTimer);
    if (tipFor) tipFor.removeAttribute("aria-describedby");
    tipFor = null;
    if (tipEl) tipEl.classList.remove("on");
  }
  function tips(root) {
    (root || document).querySelectorAll("button[title], .btn[title], a[title]").forEach((el) => {
      el.setAttribute("data-tip", el.getAttribute("title"));
      el.removeAttribute("title");
    });
  }
  document.addEventListener("pointerover", (e) => {
    const t = e.target.closest && e.target.closest("[data-tip]");
    if (!t || t === tipFor) return;
    hideTip();
    if (e.pointerType === "touch") return;     // touch never depends on hover
    tipTimer = setTimeout(() => showTip(t), 180);
  });
  document.addEventListener("pointerout", (e) => {
    const t = e.target.closest && e.target.closest("[data-tip]");
    if (t && !t.contains(e.relatedTarget)) hideTip();
  });
  document.addEventListener("focusin", (e) => {
    const t = e.target.closest && e.target.closest("[data-tip]");
    hideTip();
    if (t && t.matches(":focus-visible")) showTip(t);
  });
  document.addEventListener("focusout", hideTip);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") hideTip(); }, true);
  window.addEventListener("scroll", hideTip, true);

  // ----------------------------------------------------------- timeline
  // Lead playback in exact steps. The thumb moves between real lead days and
  // nothing is drawn in between: there is no value at Day 4.5, so the
  // animation never implies one. Longest lead on the left, as on the landing:
  // left to right is time approaching the event.
  //
  //   opts.get()          current lead
  //   opts.set(L)         show lead L (the page renders it)
  //   opts.validDate(L)   the row's own valid date, or null
  function timeline(host, opts) {
    const SPEEDS = [0.5, 1, 2];
    let timer = null, dir = -1, speed = 1, startLead = null;
    host.classList.add("timeline");
    host.setAttribute("role", "group");
    host.setAttribute("aria-label", "Lead day playback");
    host.innerHTML =
      `<button type="button" class="tl-play" aria-pressed="false" data-tip="Play the lead days in order (space)">Play</button>` +
      `<div class="steps" role="group" aria-label="Lead day">` +
      [10, 9, 8, 7, 6, 5, 4, 3, 2, 1].map((L) =>
        `<button type="button" data-lead="${L}" aria-pressed="false">${code(L)}</button>`).join("") +
      `</div>` +
      `<button type="button" class="tl-dir" data-tip="Direction of playback">Toward<span class="tl-long"> the event</span></button>` +
      `<button type="button" class="tl-speed" data-tip="Playback speed">1×</button>` +
      `<button type="button" class="tl-reset" data-tip="Stop and return to the lead shown before playing">Reset</button>` +
      `<span class="valid" aria-live="off"></span>`;
    const play = host.querySelector(".tl-play"), dirB = host.querySelector(".tl-dir");
    const speedB = host.querySelector(".tl-speed"), reset = host.querySelector(".tl-reset");
    const valid = host.querySelector(".valid");

    function refresh() {
      const L = opts.get();
      host.querySelectorAll(".steps button").forEach((b) => {
        const n = +b.dataset.lead, vd = opts.validDate ? opts.validDate(n) : null;
        b.setAttribute("aria-pressed", String(n === L));
        b.setAttribute("aria-label", `Day ${n}` + (vd ? `, valid ${vd}` : ""));
      });
      const vd = opts.validDate ? opts.validDate(L) : null;
      valid.textContent = `${code(L)} · valid ${vd || "n/a"}`;
      play.textContent = timer ? "Pause" : "Play";
      play.setAttribute("aria-pressed", String(!!timer));
      dirB.innerHTML = dir < 0 ? `Toward<span class="tl-long"> the event</span>`
                               : `Away<span class="tl-long"> from the event</span>`;
      speedB.textContent = speed + "×";
    }
    function stop() { clearTimeout(timer); timer = null; refresh(); }
    function tick() {
      const next = opts.get() + dir;
      if (next < 1 || next > 10) { stop(); return; }
      opts.set(next);
      timer = setTimeout(tick, 760 / speed);
      refresh();
    }
    function start() {
      startLead = opts.get();
      // At the end of the run, play starts again from the far end.
      if ((dir < 0 && startLead === 1) || (dir > 0 && startLead === 10)) opts.set(dir < 0 ? 10 : 1);
      timer = setTimeout(tick, 760 / speed);
      refresh();
    }
    function toggle() { if (timer) stop(); else start(); }

    play.onclick = toggle;
    dirB.onclick = () => { dir = -dir; refresh(); };
    speedB.onclick = () => { speed = SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length]; refresh(); };
    reset.onclick = () => { const back = startLead; stop(); if (back) opts.set(back); refresh(); };
    host.querySelector(".steps").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-lead]");
      if (!b) return;
      stop(); opts.set(+b.dataset.lead); refresh();
    });
    // A hidden tab does not keep stepping through leads nobody is watching.
    document.addEventListener("visibilitychange", () => { if (document.hidden && timer) stop(); });
    refresh();
    return { refresh, stop, toggle, get playing() { return !!timer; } };
  }

  // --------------------------------------------------------- provenance
  // "Why this number?": every field the row and /api/metrics actually carry,
  // and "unavailable" for the rest. Nothing is filled in by guessing.
  function provenance(r, ctx) {
    ctx = ctx || {};
    const served = ctx.served || null;
    const na = (t) => `<dd class="na">${esc(t || "not provided by this build")}</dd>`;
    const dd = (v) => (v == null || v === "" ? na() : `<dd>${esc(v)}</dd>`);
    const ok = r.status === "OK";
    const combined = typeof r.model_probability === "number";
    const rows = [
      ["Served number", ok ? (combined ? `${pct(r.bust_probability)}, the model + ensemble combination`
                                       : `${pct(r.bust_probability)}, the model alone`)
                           : "none: the model declined to score this row"],
      ["Status", `${r.status}` + (r.review_tier ? ` · tier ${r.review_tier}` : "")],
      ["What the tier means", r.tier_guidance],
      ["Model alone", pct(r.model_probability)],
      ["Ensemble spread", typeof r.ens_spread_mm === "number" ? `${r.ens_spread_mm.toFixed(1)} mm/day (50 members)` : null],
      ["Spread baseline", pct(r.baseline_probability)],
      ["Interval", Array.isArray(r.prediction_interval)
        ? `${pct(r.prediction_interval[0])} to ${pct(r.prediction_interval[1])}` : null],
      ["Confidence in estimate", pct(r.confidence_in_estimate)],
      ["Distance from training", typeof r.ood_distance === "number" ? r.ood_distance.toFixed(2) + " (refusal detector)" : null],
      ["Issued", r.init_date],
      ["Valid", r.valid_date],
      ["Lead", `Day ${r.lead_day}`],
      ["Store version", r.model_version],
      ["Base model", served && served.base_model],
      ["Reasons from", served && served.reasons_method],
      ["Refusal checks", served && served.ood_inputs],
      ["Clock", ctx.mode === "replay"
        ? "Archive replay: this issue date is treated as now. Not a live feed."
        : ctx.mode],
      ["Data quality", r.data_quality],
      ["Input age", typeof r.input_age_hours === "number"
        ? `${r.input_age_hours.toFixed(0)} h` + (ctx.mode === "replay" ? " (nominal for a 00Z run in replay)" : "") : null],
      ["Source", ctx.source],
    ];
    return `<details class="prov"><summary>Why this number?</summary><dl>` +
      rows.map(([k, v]) => `<dt>${esc(k)}</dt>` + dd(v)).join("") + `</dl></details>`;
  }

  // The clock badge every working view shows: replay is never presented as live.
  function mode(el, first, last) {
    if (!el) return;
    el.innerHTML = `Archive replay<span class="long"> · ${esc(first)} to ${esc(last)}</span>`;
    el.setAttribute("data-tip", `Historical archive, ${first} to ${last}. Each issue date is replayed as if it were today; nothing on this screen is live.`);
  }

  read();
  return { read, write, relink, query, onRemote, tips, timeline, provenance, mode, esc, pct, code, REDUCED,
           get state() { return { ...state }; } };
})();

document.addEventListener("DOMContentLoaded", () => { FBI.relink(); FBI.tips(); });
