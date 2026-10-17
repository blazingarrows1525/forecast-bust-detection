---
name: Forecast Bust Intelligence
description: Plots the reliability of issued rainfall forecasts over India like a synoptic chart and flags the ones likely to bust.
colors:
  gauge-periwinkle: "#8fa6e8"
  gauge-indigo: "#27407a"
  accent-ink-orbit: "#0d1430"
  accent-ink-slate: "#0e1530"
  accent-ink-paper: "#ffffff"
  baseline-ochre: "#d9a24a"
  baseline-ochre-paper: "#8d5a12"
  outcome-rust: "#e0a274"
  outcome-rust-paper: "#7a4126"
  focus-pale: "#c3d0f5"
  risk-c0: "#440154"
  risk-c1: "#414487"
  risk-c2: "#2a788e"
  risk-c3: "#5ec962"
  risk-c4: "#fde725"
  orbit-void: "#100e19"
  orbit-void-2: "#16141f"
  orbit-plate: "#1e1b2a"
  orbit-plate-2: "#272435"
  plotting-ink: "#ecebf4"
  plotting-ink-2: "#b7b4c9"
  plotting-ink-3: "#9390a8"
  orbit-rule: "#25212f"
  orbit-rule-2: "#3a3550"
  orbit-hatch-ink: "#e4e2ee"
  orbit-hatch-bg: "#2b2738"
  globe-ocean: "#141633"
  globe-deep: "#05040e"
  globe-graticule: "#363a6c"
  globe-limb: "#6d80e4"
  land: "#2d2a3f"
  land-no-reading: "#3d3a52"
  slate-ground: "#15191c"
  slate-ground-2: "#1b2024"
  slate-plate: "#20262b"
  slate-plate-2: "#283036"
  slate-ink: "#e6e9e4"
  slate-ink-2: "#adb5b9"
  slate-ink-3: "#8f989e"
  slate-rule: "#2f363c"
  slate-rule-2: "#414a51"
  chart-paper: "#f2f4f0"
  chart-paper-2: "#e8ebe5"
  enamel-plate: "#fbfcf9"
  enamel-plate-2: "#eceee8"
  chart-ink: "#172133"
  chart-ink-2: "#465163"
  chart-ink-3: "#5b6574"
  graticule: "#c9cec5"
  graticule-2: "#a9afa6"
  warn-bg-orbit: "#33270f"
  warn-bg-slate: "#3a2c12"
  warn-rule-dark: "#7a5a22"
  warn-ink-dark: "#f1d9a6"
  warn-bg-paper: "#f6ecd6"
  warn-rule-paper: "#c9a35a"
  warn-ink-paper: "#4a3410"
typography:
  display:
    fontFamily: "Bahnschrift, \"DIN Alternate\", \"DIN 2014\", \"Roboto Condensed\", \"Arial Narrow\", system-ui, sans-serif"
    fontSize: "clamp(40px, 4.4vw, 66px)"
    fontWeight: 600
    lineHeight: 1.03
    letterSpacing: "-0.016em"
  headline:
    fontFamily: "Bahnschrift, \"DIN Alternate\", \"DIN 2014\", \"Roboto Condensed\", \"Arial Narrow\", system-ui, sans-serif"
    fontSize: "clamp(26px, 2.6vw, 36px)"
    fontWeight: 600
    lineHeight: 1.12
    letterSpacing: "-0.008em"
  title:
    fontFamily: "Bahnschrift, \"DIN Alternate\", \"DIN 2014\", \"Roboto Condensed\", \"Arial Narrow\", system-ui, sans-serif"
    fontSize: "18px"
    fontWeight: 600
    lineHeight: 1.3
  reading:
    fontFamily: "Bahnschrift, \"DIN Alternate\", \"DIN 2014\", \"Roboto Condensed\", \"Arial Narrow\", system-ui, sans-serif"
    fontSize: "clamp(30px, 3.2vw, 42px)"
    fontWeight: 600
    lineHeight: 1
    fontFeature: "\"tnum\""
  station-served:
    fontFamily: "Bahnschrift, \"DIN Alternate\", \"DIN 2014\", \"Roboto Condensed\", \"Arial Narrow\", system-ui, sans-serif"
    fontSize: "19px"
    fontWeight: 600
    lineHeight: 1.1
    fontFeature: "\"tnum\""
  station-field:
    fontFamily: "Bahnschrift, \"DIN Alternate\", \"DIN 2014\", \"Roboto Condensed\", \"Arial Narrow\", system-ui, sans-serif"
    fontSize: "12.5px"
    fontWeight: 600
    lineHeight: 1.3
    fontFeature: "\"tnum\""
  lede:
    fontFamily: "system-ui, -apple-system, \"Segoe UI\", Roboto, sans-serif"
    fontSize: "clamp(17px, 1.4vw, 19.5px)"
    fontWeight: 400
    lineHeight: 1.55
  body:
    fontFamily: "system-ui, -apple-system, \"Segoe UI\", Roboto, sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.62
  body-ops:
    fontFamily: "system-ui, -apple-system, \"Segoe UI\", Roboto, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.5
  control:
    fontFamily: "system-ui, -apple-system, \"Segoe UI\", Roboto, sans-serif"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.3
  wordmark:
    fontFamily: "Bahnschrift, \"DIN Alternate\", \"DIN 2014\", \"Roboto Condensed\", \"Arial Narrow\", system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 600
    letterSpacing: "0.01em"
  label:
    fontFamily: "Bahnschrift, \"DIN Alternate\", \"DIN 2014\", \"Roboto Condensed\", \"Arial Narrow\", system-ui, sans-serif"
    fontSize: "12.5px"
    fontWeight: 600
    letterSpacing: "0.02em"
  mono:
    fontFamily: "ui-monospace, \"Cascadia Mono\", Consolas, monospace"
    fontSize: "11px"
    lineHeight: 1
rounded:
  r: "3px"
  swatch: "2px"
  station: "0px"
spacing:
  shell-y: "8px"
  shell-x: "14px"
  panel-y: "14px"
  panel-x: "16px"
  gutter: "24px"
  gutter-narrow: "16px"
  section: "92px"
  section-narrow: "64px"
  topbar: "60px"
components:
  button:
    backgroundColor: "{colors.orbit-plate}"
    textColor: "{colors.plotting-ink}"
    rounded: "{rounded.r}"
    padding: "5px 10px"
    height: "30px"
  button-primary:
    backgroundColor: "{colors.gauge-periwinkle}"
    textColor: "{colors.accent-ink-orbit}"
    rounded: "{rounded.r}"
    padding: "10px 20px"
    height: "46px"
  button-slate:
    backgroundColor: "{colors.slate-plate}"
    textColor: "{colors.slate-ink}"
    rounded: "{rounded.r}"
    padding: "5px 10px"
    height: "30px"
  lead-rail-key:
    backgroundColor: "{colors.orbit-plate}"
    textColor: "{colors.plotting-ink}"
    typography: "{typography.label}"
    rounded: "{rounded.r}"
    padding: "4px 0"
    height: "32px"
  select:
    backgroundColor: "{colors.slate-plate}"
    textColor: "{colors.slate-ink}"
    rounded: "{rounded.r}"
    padding: "4px 8px"
    height: "30px"
  title-block:
    backgroundColor: "{colors.orbit-void-2}"
    textColor: "{colors.plotting-ink-2}"
    rounded: "{rounded.r}"
    padding: "12px 14px"
    width: "408px"
  tooltip:
    backgroundColor: "{colors.orbit-plate}"
    textColor: "{colors.plotting-ink}"
    rounded: "{rounded.r}"
    padding: "8px 11px"
  station-plot:
    textColor: "{colors.plotting-ink}"
    typography: "{typography.station-served}"
    rounded: "{rounded.station}"
    padding: "4px 6px 5px"
  view-tab:
    textColor: "{colors.slate-ink-2}"
    rounded: "{rounded.r}"
    padding: "6px 9px"
  kbd:
    backgroundColor: "{colors.slate-plate}"
    textColor: "{colors.slate-ink}"
    typography: "{typography.mono}"
    rounded: "{rounded.r}"
    padding: "2px 5px"
---

# Design System: Forecast Bust Intelligence

This file describes the system as built. The shared tokens live in `web/fbi.css` as CSS custom properties (`--ground`, `--ground-2`, `--plate`, `--plate-2`, `--ink`..`--ink-3`, `--rule`, `--rule-2`, `--accent`, `--accent-ink`, `--accent-wash`, `--base`, `--outcome`, `--hatch-ink`, `--hatch-bg`, `--focus`, `--warn-*`, `--water`, `--edge-hi`, `--edge-lo`, `--shade`, `--shade-in`, `--font-sign`, `--font-text`, `--font-mono`, `--r`), re-valued per material by `:root[data-world="orbit"]`, `:root` / `[data-theme="dark"]` and `[data-theme="light"]`. The hex values above are those properties resolved per material; the globe and land colours are page-local to `web/landing.html`, and the five risk stops are declared first in each page's own `:root{` block.

## Overview

**Creative North Star: "The Night Chart"**

Forecast reliability is plotted like a synoptic chart over the real planet. India is raised on the model's own 0.25-degree grid over a plotting globe in a graphite-indigo void, every column the served reading, and each subdivision is a plotted station whose fixed positions carry the served reading, the baseline, the lead and the observed outcome. The system refuses the dark dashboard of stat cards around a glowing map, and the light editorial page the landing used to be.

One world, four materials, each with one job. **Night orbit** is the landing in every OS colour scheme: the void, the globe, the raised land and cool-white plotting ink. **Wet-concrete slate** is the ops shell's default, calm and compact, for the map, columns and volume views. **Chart paper** is the ops shell's light theme, one attribute away. **Machined plate** is every control on every material: solid, lit from above, pressed in when used.

Density follows the material. The landing is a full-bleed scene and then a long ruled read with generous rhythm; the ops shell is compact (14px body, 12-13px labels, hairline-divided panels). Everything is system fonts and same-origin; the scene is rendered locally in WebGL with a 2-D canvas fallback.

**Key Characteristics:**
- Four materials, one job each: night orbit persuades, slate operates, chart paper is the light ops theme, machined plate is every control.
- Gauge periwinkle is the only accent on dark (gauge indigo on paper); risk lives only on the five viridis stops; refusal is a hatch, never a hue.
- Station plots: unboxed ink in fixed positions, joined to their station by a leader.
- Stencilled DIN signage (Bahnschrift) for codes, readings and headings; system-ui for prose.
- Lit, not flat: controls carry an edge highlight, a darker lower edge and a short ground-tinted shade; floating plates a defined edge and a short drop. No glass, no glow.
- Whole units: the lead rail steps whole days and lead changes swap values without tweening.

## Colors

A cool graphite-indigo void and two ops grounds carrying one periwinkle accent, one ochre data series and a colour-vision-tested viridis ramp for risk.

### Primary
- **Gauge Periwinkle** (`gauge-periwinkle`; `--accent` on night orbit and slate): the primary action, the second line of the headline ("Which one?"), pressed toggles and their underline, the current-view underline, the selected-row bar, links, selection, station hover rule, and the busy hairline. Its text partner is `accent-ink-orbit` / `accent-ink-slate`. A wash (`--accent-wash`, periwinkle at 13% on orbit, 15% on slate) marks hover and the live row.
- **Gauge Indigo** (`gauge-indigo`; `--accent` on chart paper): the same role on the light theme, with `accent-ink-paper`.

### Secondary
- **Baseline Ochre** (`baseline-ochre` on dark, `baseline-ochre-paper` on paper; `--base`): the ensemble-spread baseline and only that. It always travels with a shape: a dashed line (6 5 on the meteogram), a notch on interval scales, and the fixed lower-left position in a station plot.

### Tertiary
- **Outcome Rust** (`outcome-rust` / `outcome-rust-paper`; `--outcome`): the observed-bust fill on the map's Observed view, where a fill needs a colour. Everywhere else an outcome is carried by ink weight.
- **Risk Ramp** (`risk-c0`..`risk-c4`; `--c0`..`--c4`): viridis, five stops, the only encoding of risk: the raised columns on the globe, the station marks, the meteogram, the legend.

### Neutral
- **Graphite-Indigo Void** (`orbit-void`, `orbit-void-2`): the landing ground, never pure black; the lifted void backs the title block and the top-bar scrim.
- **Night Plate** (`orbit-plate`, `orbit-plate-2`): controls, tooltips and the key plot on the landing.
- **Plotting Ink** (`plotting-ink`, `-2`, `-3`): headings and readings, prose and fields, captions and secondary labels on the landing.
- **Night Rule** (`orbit-rule`, `orbit-rule-2`): section hairlines and control borders on the landing.
- **Plotting Globe** (`globe-ocean`, `globe-deep`, `globe-graticule`, `globe-limb`): the ocean sphere shading to its dark side, the 10-degree graticule and the thin atmospheric limb. `land` is the raised grid's base, `land-no-reading` a cell with no served value.
- **Wet Slate** (`slate-ground`..`slate-rule-2`): the ops shell's default ground, panels, controls, ink and hairlines.
- **Chart Paper** (`chart-paper`..`graticule-2`): the ops shell's light theme.
- **Focus** (`focus-pale` on dark; gauge indigo on paper): the 2px focus outline.
- **Warning** (`warn-*`): the stale-data banner and the override warning only; an ochre-brown note, not an alarm.

### Named Rules
**The Hatch-Not-Hue Rule.** Refusal is a 45-degree hatch of `--hatch-ink` on `--hatch-bg` (2px ink, 3px ground), never a colour token and never a point on the risk ramp, on every material: legend swatch, table cell, map pattern, column texture.

**The One Ramp Rule.** Risk is drawn only with the five viridis stops, and nothing else uses them. No red, amber or green danger paint.

**The Single Accent Rule.** Periwinkle (indigo on paper) is the only accent. Ochre is a data series always doubled by shape; rust is only the Observed-view fill.

**The Doubled Baseline Rule.** The baseline is never ochre alone: it is dashed as a line, notched on a scale, and placed lower left in a station plot.

## Typography

**Display Font:** Bahnschrift (with "DIN Alternate", "DIN 2014", "Roboto Condensed", "Arial Narrow", system-ui)
**Body Font:** system-ui (with -apple-system, "Segoe UI", Roboto)
**Label/Mono Font:** ui-monospace (with "Cascadia Mono", Consolas), for keys and the source endpoint only

**Character:** Stencilled DIN signage, the lettering of a plotted chart and a gauge staff, against a plain system text face. Signage carries everything read off a scale; the system face carries prose.

### Hierarchy
- **Display** (600, clamp(40px, 4.4vw, 66px), 1.03, -0.016em, balanced): the landing headline only, max 11.5em; on short laptops clamp(36px, 3.7vw, 54px), on narrow screens clamp(34px, 8.6vw, 52px).
- **Headline** (600, clamp(26px, 2.6vw, 36px), 1.12): landing section heads.
- **Title** (600, 18px, 1.3): register entries, route stops, the key plot heading, the selected subdivision on ops panels.
- **Reading** (600, clamp(30px, 3.2vw, 42px), 1, tabular): measured values in the register.
- **Station served** (600, 19px, tabular): the served reading in a station plot; its fields (lead, baseline, outcome, name) are 12-12.5px.
- **Lede** (400, clamp(17px, 1.4vw, 19.5px), 1.55, max 33ch).
- **Body** (400, 16px, 1.62, max 64ch): landing prose in ink-2, emphasis in ink. **Body Ops** (14px, 1.5) on the working views; tables 14px tabular with 12.5px heads.
- **Label** (600, 12.5px, 0.02em, sentence case): title block, provenance summary, panel heads.
- **Mono** (11px): keyboard keys and the cited endpoint.

### Named Rules
**The Signage Numerals Rule.** Subdivision codes, lead days, percentages, dates in the title block, scale numerals and headings are set in the signage face with tabular numerals. Prose never is.

**The System Fonts Rule.** No `@font-face`, no webfont, no external origin. Off Windows the signage falls back to its nearest cousin. A self-hosted DIN-class face is an open owner question, not a decision.

## Layout

**Landing first viewport (night orbit):** a full-bleed scene at 100dvh. The WebGL canvas fills the hero; on screens 960px and wider the camera is offset so India sits right of centre and a falloff from the void (solid to 30%, clear by 47%) keeps the left column at full contrast. The top bar (60px) floats over the scene on a scrim, not a panel. The text column sits at 60px + 17vh (7vh on short screens) in a 1240px centred wrap with 24px gutters: headline, lede, two actions. The title block sits bottom left under the actions, 24px from the edges, max 408px; Skip intro sits bottom right while the intro runs. Station plots float in the void above India, right of the headline, as many as the band holds. Below 960px the hero stacks: copy, then the scene at 62svh (min 360px), then the title block in flow.

**Landing long read:** sections of 92px padding (64px under 640px) separated by hairlines; prose to 64ch. The case section is a 1.25fr / .75fr grid (one column under 1024px) holding the meteogram (440px tall, 380px on phones) and the station key.

**Ops shell (slate / chart paper):** a full-height grid with the stage filling the remaining width and a fixed right panel (384px on the map, 360px on the columns) scrolling independently; a top bar of wordmark, reading controls and the view nav at 8px 14px; panel sections 14px 16px divided by hairlines. Under 900px the stage goes on top and the panel follows.

**Context travels in the URL.** Date, lead and region ride in the query string; every carried link keeps them, so the landing's primary action opens the command centre on the plotted reading.

### Named Rules
**The Whole Units Rule.** The lead rail is ten equal keys, Day 1 to Day 10, and never shows a value between days; grids step in whole cells.

**The Rules-Not-Boxes Rule.** Lists of facts are hairline-ruled rows (the register, the three bust conditions joined by "and", the route, the provenance list, tables), never cards or tiles.

## Elevation & Depth

Lit, short and tinted. Depth comes from four tonal steps (ground, lifted ground, plate, plate-2) separated by one-pixel rules, and from a single light source above: every control carries a 1px top highlight (`--edge-hi`), a darker 1px lower edge (`--edge-lo`) and a short shade tinted to the ground (`--shade`), never black and never wide. Floating plates take a defined edge plus a short drop. The scene's only light is its own: one low key light on the land and the atmospheric limb.

### Shadow Vocabulary
- **Plate at rest** (`box-shadow: inset 0 1px 0 var(--edge-hi), 0 1px 0 var(--edge-lo), 0 2px 5px -2px var(--shade)`): every button.
- **Plate lifted** (`inset 0 1px 0 var(--edge-hi), 0 2px 0 var(--edge-lo), 0 7px 14px -6px var(--shade)` with `translateY(-1px)`): hover.
- **Plate sunk** (`inset 0 2px 3px var(--shade-in), inset 0 -1px 0 var(--edge-hi)` with `translateY(1px)`): active.
- **Pressed toggle** (`inset 0 2px 3px var(--shade-in), inset 0 -2px 0 var(--accent)` on the accent wash): a toggle that stays down.
- **Primary plate** (`inset 0 1px 0 rgba(255,255,255,.35), 0 1px 0 var(--edge-lo), 0 3px 8px -3px var(--shade)`): the accent action; hover brightens 8% and deepens the drop.
- **Select** (`inset 0 1px 0 var(--edge-hi), 0 1px 0 var(--edge-lo)`): no drop.
- **Floating plate** (`inset 0 1px 0 var(--edge-hi), 0 6px 10px -6px var(--shade)`; title block `0 8px 14px -8px`): tooltips, the title block, the timeline strip.
- **Selection bar** (`inset 3px 0 0 var(--accent)`): the selected row in a queue or list.
- **Swatch hairline** (`inset 0 0 0 1px rgba(127,127,127,.35)`): legend swatches and station marks.

### Named Rules
**The Short Shade Rule.** A drop is at most 0 8px 14px with a negative spread, tinted to the ground. No wide diffuse shadow, no blur behind a plate, no glow.

**The Flat-When-Disabled Rule.** Disabled is flat (no shade, no lift, 50% opacity); busy keeps the plate and runs a 2px accent hairline along its lower edge (1.1s loop, static under reduced motion).

**The Level Bar Rule.** A segmented bar is one plate: its keys sink when pressed but never lift on hover.

## Shapes

Nearly square. Controls, panels, tooltips and the title block share one small radius (3px); swatches use 2px; station plots are unboxed (0) with only a 1px rule along their top. Segmented bars fuse by dropping inner radii and overlapping borders a pixel. Keys are keycaps with a 2px bottom border. The recurring silhouettes are the raised grid column (cells meeting edge to edge, no gaps) and the station plot: a round risk mark with four fields around it and a leader line to its station.

## Components

### Buttons
Solid machined plates that respond like keys.
- **Shape:** gently squared (3px).
- **Default:** plate, ink text, 1px `--rule-2` border, 30px min height (44px on coarse pointers), 5px 10px, 13px text, plate-at-rest shading.
- **Hover / Active / Focus:** border to `--ink-3`, lift 1px (140ms, cubic-bezier(.2,.7,.2,1)); active sinks 1px in 50ms; focus is a 2px `--focus` outline offset 2px.
- **Pressed toggle:** accent wash, accent border, stays sunk with the accent underline.
- **Primary:** accent fill, `--accent-ink` text, 600. On the landing hero both actions grow to 46px, 10px 20px, 15.5px.
- **Disabled / Busy:** see Elevation & Depth.

### Chips
- **Style:** legend swatches 14x10 (2px radius) with the swatch hairline; the refused swatch is the hatch; "no reading" is the land-no-reading tone.

### Cards / Containers
- **Corner Style:** 3px.
- **Background:** the lifted ground (`--ground-2`) for the title block, panels and timeline; plate for tooltips and the key plot.
- **Shadow Strategy:** floating plate (see Elevation & Depth); panels in the ops shell sit in flow without a drop.
- **Border:** 1px `--rule-2` on floating plates, `--rule` between panel sections.
- **Internal Padding:** 12px 14px title block, 8px 11px tooltip, 14px 16px panel sections, 6px 8px timeline.

### Inputs / Fields
- **Style:** plate, 1px `--rule-2` border, 3px, top highlight and lower edge; selects 30px with 4px 8px. Range inputs take the accent; carets are the accent.
- **Focus:** the 2px focus outline.
- **Error / Warning:** the override warning is a warm note (`--warn-bg`, `--warn-rule`, `--warn-ink`).

### Navigation
- **Style:** the wordmark (signage, 15px, 600) and the view list (13px, `--ink-2`, 6px 9px).
- **States:** hover lifts the text to ink on the accent wash; the current view carries a 2px accent underline.
- **Landing:** the bar floats on a void scrim over the scene. **Mobile:** the bar stays on one row on the landing (12.5px links); on the ops shell it wraps.

### Station Plot (signature)
Unboxed ink on the chart, read in fixed positions around a 12px round risk mark: served reading upper left (signage 19px, 600), lead upper right (12.5px, ink-2), baseline lower left in ochre, observed outcome lower right (bold ink "busted", regular ink-2 "held"), subdivision name beneath (12.5px, up to two lines). A refused station's mark is the hatch and its served field reads "refused"; the baseline field is labelled "base" as well as ochre. A 1px `--ink-3` rule runs along its top (accent on hover). A ground-coloured text halo keeps every figure legible over the limb or the stars. Each plot sits in the void above the land with a 1.2px `--ink-2` leader to a 1.8px ink ring at its station. The station key in the case section draws the same convention at 30px on a plate.

### Raised Grid Globe (signature)
A plotting sphere (ocean, 10-degree graticule, atmospheric limb) with India alone raised on the model's 0.25-degree grid, each column's height and colour the served reading, cells meeting edge to edge. Hover outlines a subdivision and shows a floating tooltip; drag turns the globe with inertia. Without WebGL a 2-D canvas draws the same reading flat.

### Title Block and Lead Rail (signature)
The chart's title block: "Archive replay" and the intro phase, issue / valid / lead dates in signage, the lead rail, the legend and the cited endpoint. The rail is a segmented bar of ten whole-day keys (32px, signage, tabular) plus Play; the current day is the pressed key. A lead change swaps every column and station to the archive's own values without tweening.

### Meteogram
The ten forecasts for one day as plotted stations on a time axis, longest lead left: served reading as the station, the baseline as a dashed ochre line (6 5), the review level as a dotted ink line (2 3), outcome by ink weight; each station links into the command centre at that lead. Labels carry a ground stroke halo.

### Provenance ("Why this number?")
A disclosure under the reasons: a signage summary (12.5px) opening a ruled definition list of issue date, valid date, lead and source endpoint; missing values are italic ink-3.

### Motion
One authored intro of about 4.6 s in named stops: orbit (0-2.4 s), outline (coast drawn from 1.0 s), boundaries (2.3-2.9 s), plotted (columns rise 2.6-4.4 s). The headline, lede, actions and title block arrive with it (0.8 s, staggered .12 s / .24 s / .45 s). Skip intro finishes it at once; under reduced motion it never runs. The arriving state exists only while a class is set that a frame or timer removes, so text is never left hidden. The meteogram line draws in 1.4 s, labels fade at .9 s; both are already final without JS or with reduced motion.

## Do's and Don'ts

### Do:
- **Do** set the landing on night orbit (`data-world="orbit"`) in every OS colour scheme; slate and chart paper belong to the working views and their Theme toggle.
- **Do** give every control the machined plate: edge highlight, darker lower edge, short ground-tinted shade; lift 1px on hover, sink on press, stay sunk when pressed, flat when disabled.
- **Do** float tooltips, the title block and the timeline strip on a defined 1px edge with a drop no larger than 0 8px 14px.
- **Do** plot stations as unboxed ink in the fixed positions, with a leader to the station.
- **Do** draw refusal as the hatch everywhere it appears.
- **Do** declare `--c0:#440154; --c1:#414487; --c2:#2a788e; --c3:#5ec962; --c4:#fde725` first in each page's own `:root{` block, identical across pages.
- **Do** carry the baseline by dash, notch or position as well as ochre, and outcomes by ink weight.
- **Do** step the lead rail in whole days and swap values without tweening.
- **Do** keep text visible whenever motion does not run.

### Don't:
- **Don't** add a colour token for refusal or put refusal on the risk ramp.
- **Don't** use red, amber or green danger paint.
- **Don't** introduce a second accent.
- **Don't** use glass, backdrop blur, glows or wide diffuse shadows.
- **Don't** box station plots in cards, or put facts in equal stat tiles; use ruled rows.
- **Don't** load a webfont, use `@font-face`, or reference any external origin.
- **Don't** lift a key out of a segmented bar on hover.
- **Don't** tween readings between lead days, or show a value between whole days.
- **Don't** put kickers or eyebrows above headings.
- **Don't** use glyph characters as icons; label controls with words (arrows inside `kbd` name keys, they are not icons).
