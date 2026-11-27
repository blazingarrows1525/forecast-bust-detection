# Using the views

A short guide to the working views, written for the person using them. The
research assistant reads this file to answer "how do I…" questions, so it says
what the product does today and nothing it does not.

## The views

| view | page | what it is for |
|---|---|---|
| Case | `/landing.html` | The Assam & Meghalaya case (June 2022) on the model's own grid, the method, and the measured results |
| Map | `/` | Every subdivision for one issue date and lead day, the review queue, the reasons, and the override record. The precise, 2-D view. |
| Columns | `/command.html` | All ten lead days at once, stacked over the map in 3-D: Day 1 at the ground, Day 10 at the top, with the ranked review list beside it |
| Volume | `/volume.html` | Longitude × latitude × lead day, raymarched and nearest-sampled |
| Ask | `/assistant.html` | The research assistant, when the operator has enabled it |

Every view carries the same reading in its address (`?date=…&lead=…&region=…`),
and the links between views keep it, so a reading opened on one view is the
reading opened on the next. "Ask" carries it too: the assistant then offers to
explain that reading.

## Lead-day playback: playing through the lead days (timeline)

- **Columns:** the playback strip has **Step** (one lead day in the play
  direction), **Play**, **Reverse** (play from Day 10 toward Day 1), a speed
  menu (**Slow**, **Normal**, **Fast**) and **Reset lead** (back to the lead day
  the view opened on). The strip shows the lead day on screen and its valid
  date.
- **Map:** **Scrub** (or the space bar) runs the lead days from Day 10 to Day 1.
- **Case:** the lead rail under the scene has a button per day, Day 1 to
  Day 10, and **Play**. Inside the rail the arrow keys step whole days.

Only real lead days are shown; nothing is drawn between them.

## Keyboard shortcuts

| key | Map | Columns | Volume |
|---|---|---|---|
| ← → | lead day | lead day | slice through the lead days |
| ↑ ↓ | issue date | issue date | |
| space | scrub Day 10 → Day 1 | | start or stop the fly-through |
| m / b | served / spread baseline | | |
| v | observed outcome | | |
| i | satellite imagery for the valid date (off by default) | | |
| t | light / dark | | |
| / | jump to a subdivision | | |
| r | | reset the camera | |
| s | | | switch between one slice and the full volume |
| Tab, Enter | open a review-queue row | | |
| Esc | clear the selection, or close | clear the selection | clear the hover and focus |

On the Map, **?** lists the shortcuts.

## Selecting a subdivision

- **Map:** click a subdivision, open a row in the review queue, or use the jump
  list (**/**).
- **Columns:** hover a column to outline it and see its reading, and click it
  to focus the camera on it. **Focus the top case** brings the first
  review-queue column to the centre.
- **Case:** hover a plotted cell for its reading; click it to open that
  reading on Columns. Drag to turn the scene.

## "Why this number?"

Under a selected reading on the Map and on Columns, **Why this number?** opens
the record behind it:
- its source;
- the issue date, valid date and lead day;
- the store version and build date, and the base model;
- how the served number and the reasons are made;
- the reading's distance from the training data against the refusal threshold,
  and what that check covers;
- the review tier and the data quality.

Anything the build does not provide says "unavailable", never a guess.

## Refused readings

A hatched (striped) subdivision or column is a refusal: the atmospheric state
was unlike anything in training, so the system declined to give a number. It is
unknown, elevated risk, not a low reading. Refused days busted 23.3% of the
time against 3.4% for scored ones.

## Override a flag: Dismiss flag or Escalate (Map)

On the Map, a selected reading has **Dismiss flag** and **Escalate**. Each
needs a written reason of at least three characters, which is recorded
permanently. A public, read-only deployment refuses overrides.

## Volume

**Review boundary** outlines, on the sliced day, the cells at or above the
cost-optimal review threshold. **Refusals** shows or hides the refused cells,
and **Fly-through** (or the space bar) moves through the lead days. The view
stops redrawing when nothing moves.

## Rendering on weaker devices

Columns picks a lower quality automatically on a weaker machine; `?quality=low`
forces it, and `?quality=high` asks for the full one. The Case scene has four
tiers, `?quality=high`, `balanced`, `low` and `2d`, where `2d` draws the same
cells on a flat canvas. Nothing about the data changes with the tier. If the
browser cannot start WebGL, Columns sends the reading to the Map, which shows
every value the 3-D view does, and Volume says so and links to the column view.

## The opening

The Case page opens with a short sequence over India. **Skip intro** ends it at
once. With reduced motion set in the operating system, it starts finished.
