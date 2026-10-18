---
version: 1
slug: "web-landing-html"
primary_target: "web/landing.html"
related_targets: ["web/index.html","web/command.html","web/volume.html"]
---

## Surface

Landing page (`web/landing.html`), mode Persuade. The world reaches the three Operate surfaces (`index.html`, `command.html`, `volume.html`) through the shared control material (solid machined plates with raised, pressed, selected, disabled and busy states) and the station-plot reading, never through the cinematic scene.

Audience: technical evaluators and duty forecasters, both strongly (PRODUCT.md). Job: within seconds, see that this predicts which issued forecast is about to fail; believe it through the real Assam case and registered evidence; enter the command centre on that case.

Owner brief (2026-10-17): "make them feel the website in real life"; the frontend master prompt pins a dark, atmospheric, locally rendered globe with India in focus, a 3-6 s skippable sequence (outline, boundaries, data-backed risk layer, headline, actions), solid tactile controls and no glass, and "a modern scientific instrument with cinematic art direction". Questions were waived by the owner ("just start building").

## Direction contract

THESIS: Forecast reliability plotted like a synoptic chart over the real planet: every subdivision is a plotted station whose glyph carries the served reading, the baseline, the lead and the outcome in fixed positions, plotted from the archive over a night-side Earth. Refuses the dark dashboard of stat cards around a glowing map, and its light editorial opposite (the retired chart-paper landing).

OWN-WORLD: Night orbit: a deep graphite-indigo void, never pure black; a dark ocean globe with a fine graticule and a cool atmospheric limb; India alone raised on the model's own 0.25 degree grid, lit by one low key light; risk only on the five viridis stops, rising as columns; refusal a hatch, never a hue. Station plots in cool-white plotting ink with signage numerals; one accent, gauge periwinkle; the ensemble baseline stays ochre and dashed. Controls are solid machined plates lit from above: a top highlight, a darker lower edge, a pressed state that inverts them.

STORY: The visitor watches the planet turn to India, the chart get plotted subdivision by subdivision for one real issue date, and one station stand out as the forecast about to fail. They read what a bust is, that unknown is not safe and the measured evidence with intervals, then enter the command centre on the same reading.

FIRST VIEWPORT: Full-bleed WebGL scene at 100dvh. India fills the right 60%, tilted toward the viewer, the Earth's limb curving behind it. Left column over empty void: headline "A forecast is about to fail. Which one?", one sentence, primary "Enter the command centre", secondary "Explore the verified case". Station plots tag the highest readings with leader lines. Bottom right, the chart's title block: replay label, issue and valid dates, and the lead rail (Day 1 to Day 10, play). A Skip intro control while the sequence runs.

FORM: Synoptic station-plot chart over a night-side planet; grounded list position 6 of 7; seed key 1601c50a. Fused with the brief-pinned orbit and atmosphere material, which the brief fixes and the roll does not override.

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

## Raises (from the dealt challengers)

- Cracktro scroller queue (declined): only rear layers move; the reading plane (text, station plots, the lead rail) holds dead still while it is read.
- Cutting-bench select rail (competitive, clarity): the lead rail snaps to whole days and never shows a value between them; the day you left is marked when you return.
- Cyclorama dawn (competitive, identification): the sequence stops on named states (orbit, outline, boundaries, plotted), never a continuous fade that ends nowhere.
- Memory quilt (declined): every reading carries its provenance seam (issue date, valid date, lead, source endpoint) beside it.
- Generative parametric identity (declined): one source drives the scene; the case reading chosen by the data, not a decorative seed, sets the framing and the lit station.
- Menhera pastel (declined): repeat in whole units; grids and timelines step in whole cells and whole days.

## Signature move

Plotting the chart: on entry the stations are plotted in rank order, column by column, from `/api/risk-cube`; the lead rail re-plots the same issue date at another lead day by swapping to the archive's own values, never tweening through values the model did not give.

## Translations (named)

- Brief "Earth curvature or globe": no offline basemap exists, so the globe is a plotting sphere (ocean, graticule, limb) and India is the only land drawn, because it is the only land the product has data for.
- Brief CTA labels ("ENTER COMMAND CENTRE", "EXPLORE A VERIFIED HISTORICAL CASE"): set in sentence case; the command centre is the 3-D columns view, as PRODUCT.md names it.
- Display face: system fonts only (no @font-face); Bahnschrift carries the plotted numerals and headings.
- Theme: the landing is dark in every OS scheme (night orbit); the working views keep their own theme toggle.
- Title block (finish review round 1, 2026-10-18): built bottom left, under the actions, not bottom right. India fills the right of the first viewport, so a bottom-right block would cover readings; under the actions it sits in the void the text already owns. On phones it follows the scene, below the first viewport, because the scene and the headline take the first screen.
- Station plots (round 1): unboxed ink in a row in the void above India, right of the headline, as many as the band holds (one on a phone), each with a leader to its station; the outcome (busted or held) is the lower-right position.
- Hero reading (round 1): the widest served-over-baseline gap among leads that busted, so "Which one?" is answered by a forecast that failed; the overall widest gap is still stated, with its outcome.
- Display face: Bahnschrift with platform fallbacks, per the project's system-fonts rule (PRODUCT.md, tests/test_web_pages.py). A self-hosted DIN-class face was proposed in review and is left to the owner.

## Roll record

`impeccable concept-seed --scope direction --mode persuade` (2026-10-17): seed `1601c50a`, assigned grounded position 6, synoptic station-plot chart. Grounded list by resonance: 1 geostationary full-disk image (the brief's literal reading), 2 raised-relief museum model of India, 3 monsoon onset isochrone chart, 4 Doppler radar scope, 5 ensemble plume diagram, 6 synoptic station-plot chart, 7 broadcast weather-bulletin map. Owner waived the decision page; build path code-led (no image generation available).
