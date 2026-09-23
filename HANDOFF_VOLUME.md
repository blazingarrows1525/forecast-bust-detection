# HANDOFF_VOLUME.md — geography, slice, fly-through, fallback fix

**Scope:** only the work in commits `670857d..03e496f`. This is the operator's
manual — where each piece lives, how to run it, what's solid, and what I'd
improve first if I kept going. The *rationale* is in `DECISIONS.md` D-024 and
`docs/superpowers/specs/2026-09-23-volume-geography-flythrough-design.md`;
this file is what to do next, not why.

Related docs, so you don't have to hunt:

- `docs/FRONTEND_BUILT.md` — the retrospective for every shipped piece of frontend
- `FRONTEND_LOGIC.md` — the safety invariants and the API contract
- `DECISIONS.md` D-024 — this work, with measurements
- `docs/superpowers/specs/2026-09-23-volume-geography-flythrough-design.md` — design as approved and §10 where the build departed

---

## 1. What changed, at a glance

Four commits:

| commit | what |
|---|---|
| `670857d` | `scripts/precompute_voxel_grid.py` also writes `outline`; grid bytes unchanged |
| `39139d6` | `web/volume.html`: geography, slice, hover focus, fly-through, exact traversal, real-*p* review flag |
| `c2f5974` | `web/command.html`: viridis stops, striped refusals, camera on the south side |
| `03e496f` | `DECISIONS.md` D-024, `docs/FRONTEND_BUILT.md`, both design specs |

Test count went from **169 → 192**. Fourteen of the new tests fail against the
pre-change pages; the rest are guards that pass either way.

## 2. File map

| file | lines | what to look at first |
|---|---|---|
| `web/volume.html` | 1,424 | `PRELUDE` (traversal + `voxelAlpha`), `FRAG` (edge block, colouring), `PICK_FRAG` (dominant contributor), `buildTexture` (packs A channel), `buildGround` (floor), `startFlight`/`buildFlight`/`nextSegment`/`tickFlight`, `setSlice`, `bindKeys` |
| `web/command.html` | 568 | `:root` CSS tokens (viridis stops), `refusalTexture()`, `colourFor`, the LineSegments mesh in `buildColumns`, `orbit` initial + `Reset view` |
| `scripts/precompute_voxel_grid.py` | +109 | `_outline()`: repair → 200 m simplify → 2.5 km close → exterior rings → 1 km simplify |
| `data/interim/voxel_grid.json` | 60 KB | committed; if you rebuild it, expect the grid bytes to stay the same |
| `tests/test_voxel_grid.py` | +180 | outline agreement, exact traversal, handedness, slice/pick parity, review flag, dominant-contributor pick |
| `tests/test_web_pages.py` | +30 | column view uses dashboard ramp, refusal is a stripe, camera on the south side |

## 3. Run it locally

```bash
python -m uvicorn fbd.api.app:app --app-dir src --port 8912
```

Then in a browser:

- `/volume.html` — this work's main surface
- `/command.html` — the fallback, now viridis and right way up
- `/` — dashboard (unchanged this session)

To rebuild the outline (needs geopandas + shapely; takes ~6 s):

```bash
PYTHONPATH=src python scripts/precompute_voxel_grid.py
```

Tests:

```bash
PYTHONPATH=src python -m pytest tests/ -q
```

## 4. Invariants that matter, and where they're enforced

If you change something and one of these breaks, look here first. All tests
are file-level because a shader can't assert on itself.

| invariant | guarded by |
|---|---|
| Both shaders read the same traversal, verbatim | `test_pick_and_picture_march_the_same_way` |
| No fixed-step sampling, no per-pixel jitter (would look like refusal) | `test_traversal_is_exact_per_voxel` |
| North is `−z` in the shader, `lonLatToWorld`, and the default camera | `test_north_is_minus_z_everywhere` |
| Slice is an int; whole lead days only | `test_slice_is_a_whole_lead_day` |
| While slicing, only the sliced layer can be picked; pick shares the display's uniform | `test_pick_follows_the_slice_and_shares_its_uniform` |
| Hover names the dominant contributor, not the first-hit voxel | `test_pick_names_the_voxel_that_was_seen` |
| Slice/focus dim opacity, never colour (whitening reads as higher risk on viridis) | `test_focus_and_slice_dim_opacity_never_colour` |
| Review flag is set in JS from real *p*; the shader reads a bit, not `p >= threshold` | `test_review_flag_comes_from_the_real_probability` |
| A-channel edges come from `grid.ids` alone, so they exist even in island territories with no risk data | `test_edges_come_from_the_model_grid` |
| No `Bloom`, `EffectComposer`, `UnrealBloomPass` (glow smears values across cells) | `test_no_glow_postprocessing` |
| No case numbers hardcoded in the page; captions come from `/api/*` | `test_flythrough_hardcodes_no_case_number` |
| Reduced motion → three cuts, no interpolation; captions in `aria-live` | `test_flythrough_respects_reduced_motion_and_is_announced` |
| WebGL2 failure links to `command.html` | `test_webgl2_failure_points_at_the_fallback` |
| Outline agrees with the grid within one cell, both directions | `test_outline_never_runs_through_a_cell_the_grid_does_not_cover`, `test_grid_coast_is_never_more_than_one_cell_from_the_outline` |
| Column view uses the dashboard's exact viridis stops | `test_column_view_uses_the_dashboard_ramp` |
| Column-view refusal is a stripe, not a hue or wireframe | `test_column_view_refusal_is_a_stripe_not_a_hue` |
| Column view seen from the south side | `test_column_view_is_seen_from_the_south` |
| Air-gap: only the imagery origin from `index.html`; nothing external in `volume.html`/`command.html`/`landing.html` | CI `airgap` job, `.github/workflows/ci.yml` |

## 5. What's genuinely solid

- **Handedness.** Anchored in three places (`toGrid`, `lonLatToWorld`,
  default `theta`), and a test pins all three.
- **Traversal parity.** One string, `PRELUDE`, included by both shaders. You
  cannot edit one and forget the other.
- **Picking.** Verified against pixel colour: yellow → 70.6%, olive → 31.9%.
- **Fly-through honesty.** Runs entirely off the date's `/api/*` and reports
  what actually happened, including that 2022-06-14's top flag *held*. That
  wasn't planned; it's a check the design chose over the script.
- **Air-gap.** The CI gate scans whole file contents, not just `src`/`href`,
  so a URL built in JavaScript can't slip past.
- **Outline agreement.** Measured 0.70 and 0.94 cells; bounded to 0.75 and
  1.0 in tests, with slack that's tight enough to catch drift.

## 6. What I'd improve first, second, third

Ranked by value / effort. Reasonable people could reorder.

### Would take an hour and land cleanly

1. **Add the review boundary to the volume legend.** I added a blue line and
   named it "review boundary" in the button, but the legend still shows only
   the ramp, the stripe and the "no data" swatch. Add a line swatch and a
   sentence.
2. **Explain what a full-volume pixel is.** A one-liner near the density
   slider: *"Full-volume pixels blend every value along a ray. Slice to read a
   value."* I wrote it in the handoff; users don't read handoffs.
3. **Constants at the top.** `best < 0.02` (pick dominance threshold) and
   `uSliceDim = 0.06` are meaningful magic numbers. Move beside
   `REVIEW_THRESHOLD` so they show up when someone reads the constants.
4. **A "reset view" button** on `volume.html`, matching the column view. Bind
   it to `R` and to a button, and reset `view` (not `orbit`).
5. **`test_flythrough_reduced_motion_produces_three_cuts`.** I asserted the
   reduced flag is respected; a positive test that the reduced plan has
   exactly 3 segments with `interpolates: false` would be stronger.
6. **Rebuild the voxel grid in `main` on `--force` only.** Right now it
   rebuilds every run. Not slow, but wasteful.

### Half a day, but with real payoff

7. **Landscape phone.** I never tested < 900 px. The HUD panel and the
   caption both use fixed widths; on a phone they will overlap the volume.
   Two `@media` blocks and one flex tweak.
8. **A gzipped `voxel-grid` response.** FastAPI's `JSONResponse` doesn't gzip.
   Adding `GZipMiddleware(minimum_size=1024)` would shrink 60 KB → ~15 KB,
   help slow connections, and cost nothing. It touches every endpoint though,
   so measure first.
9. **Speed control on the fly-through.** Times are hardcoded in ms. A 0.5×/1×/2×
   toggle is one integer multiplier through `nextSegment`. Useful for demos.
10. **A tiny status line on the volume view** telling you which day is on
    screen and which is under the cursor, so hover picking is less of a
    "where did that come from" experience.
11. **Refusals visualisation.** Refusals are currently drawn as achromatic
    noise inside the volume, which is correct, but they only ever appear on
    dates with a lot of them (2022-06-21 is the demo). A small "% refused
    today" indicator on the HUD would make the toggle's value obvious.

### If you want to invest a day

12. **Full-volume shading you can actually read.** The current full view
    blends along rays; a maximum-intensity projection (`max` instead of
    front-to-back) would show mass without the muddy-blend problem, and the
    slice would still be the tool for reading one value. Add it as a third
    render mode alongside "review boundary" and "refusals".
13. **Pick a colour for the review boundary that's not the accent.** Blue
    against viridis is fine but it collides with the sliceFrame's blue
    rectangle. A saturated cyan or a pure white outline may read better.
14. **Two-page volume view.** A small overview canvas (all leads at once,
    tiny) next to the main one. The user could click a lead in the overview
    to slice. This is what the fly-through does automatically; letting
    someone do it themselves is worth it.

## 7. Known small bugs and rough edges

Nothing I'd call a defect, but worth flagging so a future contributor doesn't
wonder if they're seeing something wrong:

- **First frame after date change flashes.** The volume is disposed and
  rebuilt; there's a ~50 ms black frame. Not fixable without keeping two
  textures alive.
- **The `N` label on the floor sits in front of the volume frame's rear rung.**
  At steep pitches (`phi ≈ 0.2`) they overlap. Cosmetic; two lines of
  positioning would fix it.
- **Refusal noise animates while the camera moves** (the noise depends on
  `here`, not `uTime`, but the pixel-space projection shifts). This is why
  jittering scored data is a bad idea — it would look identical. Keep it.
- **The command view's stripe texture is a CanvasTexture created lazily.** On
  the first refused column it hitches for one frame. Precompute at boot.
- **The `outlineNote` is only shown when the outline loads.** If the payload
  has an empty `outline` list (unlikely), the note stays hidden and the user
  never learns the outline is missing. Log a console warning too.
- **The fly-through's caption box max-width is `calc(100vw - 580px)`.** On a
  small window the HUD (330 px) and the readout (260 px, only when hovering)
  can push the caption past the left edge. There's a `@media (max-width: 900px)`
  branch but it's not tight enough.

## 8. Traps I hit, so you don't

- **Any per-pixel jitter turns scored data into refusal.** Refusal *is* noise.
  Don't add a scored-data jitter for anti-aliasing.
- **8-bit texture round-off.** Anything that compares *p* against a threshold
  in the shader has to know the shader sees `round(p*255)/255`. That's why
  the review flag is set in JS from the real *p* and stored as a bit.
- **First-hit picking on a translucent field feels right and is wrong.** The
  eye reads the dominant contributor; the pick has to as well, or hover
  reports faint layers.
- **Sharing a uniform object vs. copying its value** — the pick material
  reuses `material.uniforms.uSlice` and friends *by reference*. That's the
  point; a copy would let them drift. If you refactor materials, keep the
  references.
- **`Camera.setViewOffset` for one-pixel picking** is right and cheap, but
  don't forget `clearViewOffset` before the next frame. There's a comment.
- **Analytical edges inside a march**, not sampled. The march steps ~one
  voxel; a thin line inside it would alias badly. Ray-plane intersect the
  slice's top face once, texelFetch, done.
- **The design said "brighten the hovered region"; I built "dim the rest".**
  Brightening a viridis colour pushes it up the ramp — the highlight itself
  would misstate the value. Don't restore the original design without also
  changing the ramp.

## 9. What I explicitly left off

- **Deployment (sub-project E).** Blocked on a hosting choice; AWS is out.
  GHCR + a free-tier host is the zero-cost path if that's the default you
  want.
- **HawkScan.** The scan hook fires on every commit; there's no API key.
  Nothing bad happened, but the hook noise is worth silencing when you decide
  either way.
- **Mobile touch controls.** The volume view drags-orbits on pointer events,
  which works on touch, but there's no pinch-to-zoom.
- **Stripping SIH provenance** from `DATA.md`, `HANDOFF.md`, `README.md`,
  `PROJECT_BLUEPRINT.md`, `STUDY_PLAN.md`, etc. I offered earlier and you
  haven't decided. Not urgent; the *product* has none.

## 10. If you spot something odd, do this first

1. **Reload with a cache-buster** (`?v=<anything>`). The browser is aggressive
   about caching these pages.
2. **Read the console.** All non-fatal errors are silenced; a real failure
   will land here.
3. **Check the pick.** `pickAt(x, y)` in the DevTools console returns
   `{ regionIndex, lead, prob }`. If the readout looks wrong, this tells you
   whether the shader or the JS layer is at fault.
4. **Rebuild the grid** — one silent mode is `voxel_grid.json` predating a
   change to the grid or the outline logic. `precompute_voxel_grid.py` prints
   every important number.
5. **Run `test_voxel_grid.py` and `test_web_pages.py`.** They cover most of
   what matters here.

If none of that helps, `git log --follow web/volume.html` shows every touch
this session, and each commit message says what it did.
