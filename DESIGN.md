---
name: Forecast Bust Intelligence
description: Reads issued rainfall forecasts over India like a river gauge and flags the ones likely to bust.
colors:
  gauge-indigo: "#27407a"
  gauge-indigo-slate: "#8fa6e8"
  indigo-ink-on-slate: "#0e1530"
  indigo-ink-on-paper: "#ffffff"
  baseline-ochre: "#8d5a12"
  baseline-ochre-slate: "#d9a24a"
  outcome-rust: "#7a4126"
  outcome-rust-slate: "#e0a274"
  water-paper: "#c5d0e4"
  water-slate: "#33415f"
  risk-c0: "#440154"
  risk-c1: "#414487"
  risk-c2: "#2a788e"
  risk-c3: "#5ec962"
  risk-c4: "#fde725"
  chart-paper: "#f2f4f0"
  chart-paper-2: "#e8ebe5"
  enamel-plate: "#fbfcf9"
  enamel-plate-2: "#eceee8"
  chart-ink: "#172133"
  chart-ink-2: "#465163"
  chart-ink-3: "#5b6574"
  graticule: "#c9cec5"
  graticule-2: "#a9afa6"
  slate-ground: "#15191c"
  slate-ground-2: "#1b2024"
  slate-plate: "#20262b"
  slate-plate-2: "#283036"
  slate-ink: "#e6e9e4"
  slate-ink-2: "#adb5b9"
  slate-ink-3: "#8f989e"
  slate-rule: "#2f363c"
  slate-rule-2: "#414a51"
  focus-slate: "#c3d0f5"
  warn-bg-paper: "#f6ecd6"
  warn-rule-paper: "#c9a35a"
  warn-ink-paper: "#4a3410"
  warn-bg-slate: "#3a2c12"
  warn-rule-slate: "#7a5a22"
  warn-ink-slate: "#f1d9a6"
typography:
  display:
    fontFamily: "Bahnschrift, \"DIN Alternate\", \"DIN 2014\", \"Roboto Condensed\", \"Arial Narrow\", system-ui, sans-serif"
    fontSize: "clamp(38px, 3.7vw, 54px)"
    fontWeight: 600
    lineHeight: 1.04
    letterSpacing: "-0.012em"
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
  reading-ops:
    fontFamily: "Bahnschrift, \"DIN Alternate\", \"DIN 2014\", \"Roboto Condensed\", \"Arial Narrow\", system-ui, sans-serif"
    fontSize: "40px"
    fontWeight: 600
    lineHeight: 1
    fontFeature: "\"tnum\""
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
  label:
    fontFamily: "Bahnschrift, \"DIN Alternate\", \"DIN 2014\", \"Roboto Condensed\", \"Arial Narrow\", system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 600
    letterSpacing: "0.02em"
  mono:
    fontFamily: "ui-monospace, \"Cascadia Mono\", Consolas, monospace"
    fontSize: "11px"
    lineHeight: 1
rounded:
  r: "3px"
  swatch: "2px"
spacing:
  shell-y: "8px"
  shell-x: "14px"
  panel-y: "14px"
  panel-x: "16px"
  gutter: "24px"
  gutter-narrow: "16px"
  section: "84px"
  section-narrow: "60px"
components:
  button:
    backgroundColor: "{colors.slate-plate}"
    textColor: "{colors.slate-ink}"
    rounded: "{rounded.r}"
    padding: "5px 10px"
    height: "30px"
  button-primary:
    backgroundColor: "{colors.gauge-indigo}"
    textColor: "{colors.indigo-ink-on-paper}"
    rounded: "{rounded.r}"
    padding: "10px 20px"
    height: "46px"
  button-primary-slate:
    backgroundColor: "{colors.gauge-indigo-slate}"
    textColor: "{colors.indigo-ink-on-slate}"
    rounded: "{rounded.r}"
    padding: "5px 10px"
    height: "30px"
  select:
    backgroundColor: "{colors.slate-plate}"
    textColor: "{colors.slate-ink}"
    rounded: "{rounded.r}"
    padding: "4px 8px"
    height: "30px"
  view-tab:
    textColor: "{colors.slate-ink-2}"
    rounded: "{rounded.r}"
    padding: "6px 9px"
  ops-panel-section:
    backgroundColor: "{colors.slate-ground-2}"
    textColor: "{colors.slate-ink}"
    padding: "14px 16px"
  map-dock:
    backgroundColor: "{colors.slate-ground-2}"
    textColor: "{colors.slate-ink-2}"
    rounded: "{rounded.r}"
    padding: "9px 11px"
  kbd:
    backgroundColor: "{colors.slate-plate}"
    textColor: "{colors.slate-ink}"
    typography: "{typography.mono}"
    rounded: "{rounded.r}"
    padding: "2px 5px"
---

# Design System: Forecast Bust Intelligence

Rationale, the direction roll and every rule conflict are recorded in `docs/DESIGN_DECISIONS.md`; this file describes the system as built. The shared tokens live in `web/fbi.css` as CSS custom properties (`--ground`, `--plate`, `--ink`, `--rule`, `--accent`, `--base`, `--outcome`, `--water`, `--hatch-ink`, `--hatch-bg`, `--focus`, `--warn-*`, `--font-sign`, `--font-text`, `--font-mono`, `--r`), re-valued per material; the hex values above are those custom properties resolved for each material.

## Overview

**Creative North Star: "Taking a Reading"**

A forecast is read like a river gauge. Every subdivision gets a graduated staff, one mark per lead day, and the reading is taken where the served risk meets the scale. The system refuses the category default of a dark dashboard with equal stat cards around a glowing map.

One world, three materials, each with one job. **Chart paper** (cool off-white ground, blue-black ink, hairline graticules) is the ground the case is explained on: the landing, always, whatever the OS colour scheme, and the working views' light theme. **Wet-concrete slate** is the calm ops shell for working under time pressure: the map, the columns and the volume views, dark by default. **Gauge enamel** is the reading itself: white plates with E-shaped graduations in gauge indigo, the single accent, drawn on the landing staffs, the selected-reading staff on the map, and the 3-D lead staff in the columns view.

Density follows the material. The landing reads as a long, ruled document with generous rhythm; the ops shell is compact (14px body, 12-13px labels, hairline-divided panels) because a forecaster has about 45 minutes per cycle. Both are offline and system-font only; nothing loads from another origin.

**Key Characteristics:**
- Three materials, one per job: chart paper explains, slate operates, enamel reads.
- Gauge indigo is the only accent; risk lives only on the viridis ramp; refusal is a hatch, never a hue.
- Stencilled DIN signage (Bahnschrift) for codes, readings and headings; system-ui for prose.
- Flat: depth comes from ground, plate and hairline steps, never from drop shadows.
- Hairline-ruled rows instead of boxes and cards.

## Colors

A cool, low-chroma pair of grounds (chart paper and slate) carrying one indigo accent, one ochre data series and a colour-vision-tested viridis ramp for risk.

### Primary
- **Gauge Indigo** (`gauge-indigo` on chart paper, `gauge-indigo-slate` on slate; `--accent`): the E-graduations on every enamel staff, the served-reading line, the primary action, the current-view underline, pressed toggles, the selected-row bar, links and selection. Its text partner (`--accent-ink`) is `indigo-ink-on-paper` on paper and `indigo-ink-on-slate` on slate. A translucent wash (`--accent-wash`: indigo at 9% on paper, 15% on slate) marks hover and the live row.

### Secondary
- **Baseline Ochre** (`baseline-ochre` / `baseline-ochre-slate`; `--base`): the ensemble-spread baseline, and only that. It is a data-series colour, not an accent, and it always travels with a shape: a dashed line (6 5) and a notch beside the staff.

### Tertiary
- **Outcome Rust** (`outcome-rust` / `outcome-rust-slate`; `--outcome`): the observed-bust fill on the map's Observed view, where a fill needs a colour. Everywhere else an outcome is carried by ink weight.
- **Gauge Water** (`water-paper` / `water-slate`; `--water`): the translucent surface (opacity .8) that rises up an enamel staff to the served reading; the graduations show faintly through it.
- **Risk Ramp** (`risk-c0` to `risk-c4`; `--c0`..`--c4`): viridis, five stops, the only encoding of risk. Declared first in each Operate page's own `:root{` block and identical across `index.html` and `command.html`.

### Neutral
- **Chart Paper** (`chart-paper`, `chart-paper-2`; `--ground`, `--ground-2`): the landing ground and the light theme's shell.
- **Enamel Plate** (`enamel-plate`, `enamel-plate-2`; `--plate`, `--plate-2`): staff plates, controls, label backings on paper.
- **Chart Ink** (`chart-ink`, `chart-ink-2`, `chart-ink-3`; `--ink`..`--ink-3`): headings and readings, prose, captions and scale numerals.
- **Graticule** (`graticule`, `graticule-2`; `--rule`, `--rule-2`): one-pixel rules, graticule lines, control borders.
- **Wet Slate** (`slate-ground`, `slate-ground-2`, `slate-plate`, `slate-plate-2`): the ops shell ground, its bars and panels, and its controls.
- **Slate Ink** (`slate-ink`, `slate-ink-2`, `slate-ink-3`) and **Slate Rule** (`slate-rule`, `slate-rule-2`): text and hairlines on slate.
- **Warning** (`warn-*`): the stale-data banner and the dismiss-a-refusal warning only. An ochre-brown note, not an alarm.
- **Focus** (`focus-slate` on slate; gauge indigo on paper; `--focus`): the 2px focus outline.

### Named Rules
**The Hatch-Not-Hue Rule.** Refusal is a 45-degree hatch of `--hatch-ink` on `--hatch-bg` (ink on plate), never a colour token and never a point on the risk ramp. A refused case busts 23.4% of the time against 3.4% for a scored one, so it must never read as quieter than a reading. This is a deliberate encoding, the product's most important state, not decoration.

**The One Ramp Rule.** Risk is drawn only with the five viridis stops, and nothing else uses them. Green-amber-red is the warning service's palette; this product issues no warnings.

**The Single Indigo Rule.** Gauge indigo is the only accent. The baseline's ochre is a data series that is always doubled by shape, and the outcome rust appears only as the Observed-view fill. A third hue needs a data series to carry, not a mood.

**The Review-Level Rule.** No danger paint. Staffs carry a review level (0.0909) painted in ink across every staff, and the copy says "review level", never "danger level".

## Typography

**Display Font:** Bahnschrift (with "DIN Alternate", "DIN 2014", "Roboto Condensed", "Arial Narrow", system-ui)
**Body Font:** system-ui (with -apple-system, "Segoe UI", Roboto)
**Label/Mono Font:** ui-monospace (with "Cascadia Mono", Consolas), for keys only

**Character:** Stencilled DIN signage, the lettering on a gauge staff, set against a plain system text face. The signage carries everything you read off a scale; the system face carries everything you read as prose.

### Hierarchy
- **Display** (600, clamp(38px, 3.7vw, 54px), 1.04, -0.012em, balanced): the landing headline only.
- **Headline** (600, clamp(26px, 2.6vw, 36px), 1.12): landing section heads.
- **Title** (600, 18px, 1.3): register entries, route stops, the selected subdivision name on the ops panels.
- **Reading** (600, clamp(30px, 3.2vw, 42px), 1, tabular numerals): measured values in the landing register. On the ops panels the selected reading is 40px on the map and 36px in the columns view; a refused reading drops to 26px.
- **Body** (400, 16px, 1.62, max 64ch): landing prose, set in `chart-ink-2` with emphasis in `chart-ink`. The lede is clamp(17px, 1.45vw, 19px) at 34ch.
- **Body Ops** (400, 14px, 1.5): the working views. Tables are 12.5px with tabular numerals.
- **Label** (600, 12-13px, 0.02em, sentence case): control-group labels and panel section heads on the ops shell.
- **Mono** (11px, 1): keyboard keys only.

### Named Rules
**The Signage Numerals Rule.** Subdivision codes (D01-D10), lead codes, percentages, scale numerals and headings are set in the signage face with tabular numerals. Prose never is.

**The System Fonts Rule.** No `@font-face`, no webfont, no external origin. Off Windows the signage voice falls back to its nearest cousin; that is accepted.

## Layout

**Landing (chart paper):** a centred column, max 1240px, 24px gutters (16px under 640px). The first viewport is a two-column grid (1fr text, 1.18fr figure; gap 28px 44px) filling the viewport under a 56px top bar; the figure, ten staffs D10 to D01, is the dominant element at min(74vh, 620px). Below 960px it stacks to one column with the figure at 430px (400px under 480px). The long read is a sequence of sections 84px tall in padding (60px under 640px), each separated by a hairline. Prose holds to 64ch.

**Ops shell (slate):** a full-height grid: the map or 3-D stage takes the remaining width, a fixed right panel (384px on the map, 360px on the columns) scrolls independently, and a full-width top bar holds the wordmark, reading controls (date, lead, region), view toggles and the view nav. Bar padding 8px 14px; panel sections 14px 16px, divided by hairlines. Under 900px the page scrolls: the stage sits on top at about 60vh (min 320-340px) and the panel follows below it. The date slider hides below 1380px.

**Context travels in the URL.** Date, lead and region ride in the query string and every carried link keeps them, so a reading opened on one view is the reading opened on the next.

### Named Rules
**The Rules-Not-Boxes Rule.** Lists of facts are hairline-ruled rows (the measured register, the three bust conditions joined by "and", the route through the views, tables), not cards or tiles.

## Elevation & Depth

Flat. Depth is tonal: ground, a slightly lifted ground for bars and panels, and plate for controls, each step separated by a one-pixel rule. There are no drop shadows anywhere; the map tooltip explicitly removes Leaflet's. The only shadows are inset marks that carry state.

### Shadow Vocabulary
- **Selection bar** (`box-shadow: inset 3px 0 0 var(--accent)`): the selected row in a queue or list, with the accent wash behind it.
- **Pressed underline** (`box-shadow: inset 0 -2px 0 var(--accent)`): a toggle in its pressed state.
- **Swatch hairline** (`box-shadow: inset 0 0 0 1px rgba(127,127,127,.35)`): legend swatches and chips, so the palest ramp stop holds its edge on paper and on slate.

### Named Rules
**The Flat Slate Rule.** Nothing floats. Overlays on the 3-D stage and the map key sit on a lifted ground with a hairline border; the help dialog sits on a dark scrim (rgba(10,13,16,.72)), not a shadow.

## Shapes

Nearly square. Controls, panels, tooltips and the map key share one small radius (3px, `--r`); swatches, chips, staff plates and label backings use 2px. Segmented controls fuse into one bar by dropping inner radii and overlapping borders by a pixel. Keys are drawn as keycaps with a 2px bottom border. The recurring silhouette is the gauge staff: a tall narrow plate with E-shaped graduations alternating sides, one E per ten percent, the plate's own numeral beside each where the plate is wide enough.

## Components

### Buttons
Quiet, compact controls that let the reading speak.
- **Shape:** gently squared (3px).
- **Default:** plate background, ink text, 1px `--rule-2` border, 30px min height, 5px 10px padding, 13px text.
- **Hover / Focus:** the border darkens to `--ink-3` (150ms ease); pressed nudges down 1px; focus is a 2px `--focus` outline offset 2px.
- **Pressed toggle:** accent wash with an accent border and the pressed underline.
- **Primary:** gauge indigo fill, `--accent-ink` text, weight 600; hover brightens 8%. On the landing the actions grow to 46px with 10px 20px padding and 15.5px text.
- **Segmented:** adjacent buttons fused into one bar, outer corners only rounded.

### Chips
- **Style:** a 10px ramp square (2px radius) with the swatch hairline, inline before a value. Legend swatches are 16x12 (14x11 on the columns view); the refused swatch is the hatch.

### Cards / Containers
- **Corner Style:** 3px.
- **Background:** the lifted ground (`--ground-2`) on the ops shell; the landing has no containers, only ruled rows.
- **Shadow Strategy:** none (see Elevation & Depth).
- **Border:** 1px `--rule`.
- **Internal Padding:** 14px 16px for panel sections, 9px 11px for the map key, 6px 9px for stage overlays.

### Inputs / Fields
- **Style:** plate background, 1px `--rule-2` border, 3px radius; selects are 30px tall with 4px 8px padding. Reading selects (date, lead) use the signage face at 14px, weight 600, tabular. Range inputs take the accent colour.
- **Focus:** the 2px focus outline.
- **Error / Warning:** the override warning is a warm note (`--warn-bg`, `--warn-rule`, `--warn-ink`); the override reason is required and never prefilled.

### Navigation
- **Style:** the wordmark (a small indigo plate struck with one E graduation, and the product name in the signage face, 15px, 600) and the view list (13px, `--ink-2`).
- **States:** hover lifts the text to ink on the accent wash; the current view carries a 2px accent underline.
- **Mobile:** the bar wraps; on the ops shell the nav takes its own full-width row.

### Gauge Staff (signature)
One enamel plate per forecast. The served bust probability is the water level on the staff; the ink meniscus (2.6px, 4px on the live reading) marks the reading; the baseline is a dashed ochre line with a notch beside each staff; the review level is a band painted in ink across every staff; a refused lead is a hatched ghost plate at full height with no reading on it, and the served line breaks there rather than bridging it. Chart-paper graticules run behind at every ten percent. Below each staff: the lead code, then the observed outcome in ink weight (bold "bust", regular "held"). On enter the line draws (1.4s), water rises (1s, staggered 80ms per staff) and labels fade in; with reduced motion, or without JS, everything is already in its final state.

## Do's and Don'ts

### Do:
- **Do** set the landing on chart paper (`data-world="chart"`) in every OS colour scheme; dark slate belongs to the working views and their Theme toggle.
- **Do** draw refusal as the hatch (`--hatch-ink` on `--hatch-bg`) everywhere it appears: map pattern, column texture, legend swatch, ghost plate, table cell.
- **Do** declare the viridis stops `--c0:#440154; --c1:#414487; --c2:#2a788e; --c3:#5ec962; --c4:#fde725` first in each Operate page's own `:root{` block, identical across pages.
- **Do** alias page-local names to the shared tokens in `web/fbi.css` so the light theme stays one attribute away.
- **Do** set codes, readings and headings in the signage face with tabular numerals.
- **Do** carry the baseline by dash and notch as well as ochre, and carry outcomes by ink weight.
- **Do** separate content with one-pixel rules and tonal steps.
- **Do** make motion an enhancement that reduced motion and no-JS both skip.

### Don't:
- **Don't** add a colour token for refusal (no `--ood:`), or put refusal on the risk ramp.
- **Don't** use red, amber or green danger paint, or say "danger level".
- **Don't** introduce a second accent; gauge indigo is the only one.
- **Don't** load a webfont, use `@font-face`, or reference any external origin.
- **Don't** add a build step or a framework; the pages are native CSS, inline SVG and vendored scripts.
- **Don't** use drop shadows or glows for depth.
- **Don't** put a section in equal stat cards or tiles; use ruled rows.
- **Don't** put kickers or eyebrows above headings, or number items that are not a sequence.
- **Don't** use glyph characters as icons; label controls with words, and draw the few marks that are needed as ruled SVG strokes.
