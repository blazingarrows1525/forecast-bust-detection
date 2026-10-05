# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- **Duty forecaster / regional analyst (primary).** Reviews issued
  medium-range rainfall forecasts for India's 34 modelled IMD subdivisions,
  usually against a deadline (about 45 minutes per cycle), and needs to find
  the region/lead-day forecasts that deserve a careful second look, and why.
- **Technical evaluator (equally important on the landing page).** A
  reviewer, scientist or recruiter judging whether the model evidence,
  uncertainty, limitations and reproducibility hold up. Confirmed 2026-10-05:
  the landing page must convince both audiences strongly.

## Product Purpose

Forecast Bust Intelligence predicts **when an existing medium-range rainfall
forecast over India is likely to fail**: which subdivision, which lead day,
and why. It does not forecast the weather and does not replace the forecast.
It gives the forecaster a ranked review queue of region/lead cases, a map and
3-D views over the same precomputed data, and the reasons and uncertainty
behind every number. Success means the forecaster reaches the right case
faster and understands what the system says and what it does not know.

## Positioning

A forecast can look ordinary while its reliability signal deteriorates. The
product reads the model and the real 50-member ensemble *together* (the served
number is their combination) and states its own limits: it refuses to score
states unlike its training data, and those refusals bust far more often than
scored cases. Every comparative claim it makes was pre-registered before the
data was scored.

## Operating Context

- Ops-room use, often in dark environments; must run with the network
  unplugged (air-gapped). The shipped build serves a 2016–2022 reanalysis
  archive in replay mode; live mode correctly reports STALE.
- Surfaces: an evidence landing page, the 2-D operational map with the review
  queue, a 3-D command centre (risk columns over lead days), and a raymarched
  3-D risk volume (WebGL2).
- Public demo deployments run read-only (`FBD_READ_ONLY=1`).

## Capabilities and Constraints

- Served probability = model + ENS-spread combination; the model alone and the
  ENS spread are shown only as labelled components.
- Three review tiers: AUTO_OK / REVIEW / REFUSE. REFUSE means unknown,
  elevated risk, never low risk. REVIEW threshold 0.0909 is cost-derived
  (10:1 miss:false-alarm).
- Every number comes from the API; nothing is interpolated between lead days
  or regions; null stays null. Boundary cells are never blended.
- Offline-first: no external origins (fonts, CDNs, tiles, icon libraries)
  except opt-in, date-matched satellite imagery on the 2-D map.
- No build step: static HTML/CSS/vanilla JS, vendored Leaflet and Three.js r128.
- Decision support only: never issues, suppresses or implies a public warning.

## Brand Commitments

- Name on product surfaces: **Forecast Bust Intelligence**, spelled out. The
  "FBI" acronym stays in documentation only (confirmed 2026-10-05).
- Voice: precise, calm, evidence-based, not alarmist. No AI-marketing claims.
- No competition branding anywhere on the product.
- Do not reuse IMD's official warning colours; the product does not warn.

## Evidence on Hand

- The bulletin store (`bulletins.sqlite`, 79,900 rows, 2021–2022) and every
  API endpoint built on it.
- The flagship case: Assam & Meghalaya, valid 17 June 2022 (served via
  `/api/convergence`).
- Registered results D-025…D-031 (`DECISIONS.md`) and the evaluation figures
  (`docs/figures/`).
- No testimonials, users, deployments or press exist; none may be invented.

## Product Principles

1. Unknown is not safe: refusal and staleness are always visible.
2. Every number is the archive's, never a remembered or animated one.
3. The operational surfaces serve a deadline; the landing page may tell a story.
4. Show the evidence and its limits together.
5. Advise review; never direct action.

## Accessibility & Inclusion

Best effort against the brief's checklist (confirmed 2026-10-05, no formal
standard): keyboard operation of critical tasks, visible focus,
`prefers-reduced-motion`, risk and refusal readable without colour alone, a
colour-vision-deficiency-safe risk ramp (already enforced by tests), text or
list alternatives for the map and 3-D views.
