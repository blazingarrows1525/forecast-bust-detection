# Using the views

A short guide to the working views, written for the person using them. The
research assistant reads this file to answer "how do I…" questions, so it says
what the product does today and nothing it does not.

## The views

| view | page | what it is for |
|---|---|---|
| Case | `/landing.html` | The Assam & Meghalaya case (17 June 2022), the method, and the measured results |
| Map | `/` | Every subdivision for one issue date and lead day, the review queue, the reasons, and the override record. The precise, 2-D view. |
| Columns | `/command.html` | All ten lead days at once, stacked over the map in 3-D: Day 1 at the ground, Day 10 at the top |
| Volume | `/volume.html` | Longitude × latitude × lead day, raymarched and nearest-sampled |
| Ask | `/assistant.html` | The research assistant, when the operator has enabled it |

Every view carries the same reading in its address (`?date=…&lead=…&region=…`),
so a reading opened on one view is the reading opened on the next.

## Archive replay

The badge "Archive replay" means the screen is showing the 2021–2022 archive,
with each issue date treated as if it were today. Nothing on these screens is a
live feed. Hover the badge to see the archive's date range.

## Lead timeline: playing through the lead days

The strip under the header on the Map and Columns views steps through the ten
lead days, longest lead on the left:

- **Play / Pause** (or the space bar) steps one lead day at a time.
- **Toward the event / Away from the event** sets the direction.
- **1× / 2× / 0.5×** sets the speed.
- **Reset** stops and returns to the lead shown before you pressed Play.
- Selecting **D10 … D01** jumps straight to that lead.

The timeline only ever shows real lead days; nothing is drawn between them.
Playback pauses when the tab is hidden. The valid date of the lead on screen is
shown at the end of the strip.

## Keyboard shortcuts

| key | Map | Columns |
|---|---|---|
| ← → | lead day | lead day |
| ↑ ↓ | issue date | issue date |
| space | play / pause the lead days | play / pause the lead days |
| m / b / v | served / spread baseline / observed | |
| i | satellite imagery for the valid date (off by default) | |
| t | light / dark | |
| / | jump to a subdivision | |
| r | | reset the camera |
| f | | bring the top review-queue column to the centre |
| ? | all shortcuts | |
| Esc | clear the selection | clear the selection |

## Selecting a subdivision, and keeping views in step

Select a subdivision on the map, in the review queue, or with the jump list. On
Columns, select a column; hovering a column outlines it and names its reading.

When the Map and Columns are open side by side (two tabs or windows of the same
site), they follow each other: selecting a subdivision or changing the lead or
issue date on one changes the other. This uses the browser's own channel
between tabs and sends nothing over the network.

## "Why this number?"

Under a selected reading, **Why this number?** opens the record behind it: the
served number and its two ingredients (the model alone and the ensemble spread),
the spread baseline, the interval, the review tier and what it means, the
distance from training data used by the refusal detector, the issue and valid
dates, the store version, the base model, how the reasons were computed, the
clock (archive replay), data quality and the API address the row came from.
Anything the build does not provide is shown as "not provided by this build",
never guessed.

## Refused readings

A hatched (striped) subdivision or column is a refusal: the atmospheric state
was unlike anything in training, so the system declined to give a number. It is
unknown, elevated risk, not a low reading. Refused days busted 23.3% of the
time in the held-out year against 3.4% for scored ones.

## Render quality on Columns

The **Quality** menu on Columns sets the pixel density: auto (chosen from what
the device reports), high, balanced, low, or **2-D map only**, which opens the
same reading on the map. The legend panel shows the current draw calls,
triangles and pixel ratio. If the browser cannot start WebGL, the view says so
and offers the map, which shows every value the 3-D view does.

## The landing opening

The first screen of the Case page draws India, raises every subdivision's
served risk as a column from the archive, and labels one reading. Any click,
key or scroll finishes it at once; **Skip intro** does the same. With reduced
motion set in the operating system it starts finished.
