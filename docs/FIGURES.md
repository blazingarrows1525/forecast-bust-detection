# Evaluation figures

Two claims carry this project, and until now both lived only in prose and a JSON
file: that the model is **calibrated**, and that its refusals are **earned**.
Regenerate both figures with:

```bash
PYTHONPATH=src python scripts/plot_evaluation_figures.py
```

Source: `data/artifacts/bulletins.sqlite` — the same store the dashboard serves,
so these are the product's own outputs, not a separate analysis that could drift
from it.

---

## 1. Calibration — `reliability.png`

![Reliability diagram](figures/reliability.png)

A probability is only useful if it means what it says: of the days the model
calls 5%, about 5% should bust. That is the whole claim behind putting a number
in front of a forecaster, and a reliability diagram is the only honest way to
show it.

**Two things this figure is careful about.**

*Test year only.* The store also holds 2021, which is the **validation** split —
`config.VAL_YEARS` — and the isotonic calibrator was *fitted* on it. Plotting a
calibration curve on the data the calibrator was fitted to produces a beautiful
diagonal that means nothing. 2022 is touched only at final evaluation.

*The project's own binning.* Bins and ECE come from `fbd.evaluate.metrics`,
which uses **equal-count** rather than equal-width bins. With a 3.2% base rate,
equal-width bins leave the upper bins nearly empty and produce a dramatic
diagram about almost no data. Using a second definition here would put a number
on the figure that disagreed with the number in `results.json`.

**What it actually shows, including the part that is not flattering.** ECE is
0.0107 and the curve tracks the diagonal closely through the bulk of the range.
But the **highest bin predicts 0.207 and observes 0.158** — overconfident by
0.049, about 24% in relative terms. That bin is exactly where a forecaster is
looking, and it is precisely what a single low ECE hides, which is why the
figure labels it rather than leaving the summary statistic to imply the curve
sits on the diagonal everywhere. The honest statement is *"well calibrated
through the operational range, overconfident in the extreme tail, on ~80 rows."*

The lower panel uses **equal-width** bins, deliberately unlike the panel above:
the markers there each carry the same number of rows by construction, so a
histogram over those bins would be flat and say nothing. It shows how far into
the tail the model is willing to go, and on how little data.

### Cross-check against `results.json`

| | this figure | `results.json`, row `4 XGBoost + isotonic` |
|---|---|---|
| n | 19,887 | 20,060 |
| ECE | 0.0107 | 0.0108 |
| Brier | 0.0286 | 0.0291 |
| BSS | +0.072 | +0.088 |

**The gap is not an error, and it is worth understanding.** `results.json`
scores every test row in the decision band. This figure scores only the rows the
product actually *serves* — and the product refuses 173 of them as out of
distribution, so they carry no probability. 19,887 + 173 = 20,060 exactly.

Those 173 refused rows bust at **17.3%** against 3.2% for the accepted ones.
Removing them lowers the base rate of what remains, which lowers the reference
Brier score that BSS is measured against — so the served subset shows a slightly
lower skill score than the score-everything evaluation. That is the arithmetic
of refusing the hardest rows, not a discrepancy.

---

## 2. Earned refusal — `earned_refusal.png`

![Earned refusal](figures/earned_refusal.png)

The distinction a binary flag destroys. When the Mahalanobis detector puts a
region-day outside the training distribution, the system returns
`OUT_OF_DISTRIBUTION` and no probability. The question a judge should ask is
whether that refusal is a real signal or an excuse.

On the 2022 test year:

| | rows | actually busted |
|---|---|---|
| scored (`status: OK`) | 39,565 | **3.4%** |
| refused (out of distribution) | 385 | **23.4%** |

Days the model declines to score bust **6.9× more often** than the days it
accepts. A refusal is therefore not a low-risk result — it is the opposite, and
collapsing it into "not flagged" would hide the highest-risk days in the
product. This reproduces the 23.4% / 3.4% figure quoted in `LOGIC.md` 11.2
directly from the served store.

This is also why `quality/escalation.py` has three tiers rather than two, and
why the assistant's status-grounding guardrail (D-019 addendum 4) treats "the
system declined to score this" as a claim that must be backed by a tool result:
said about a scored row, it inverts the meaning of the most important cell on
the map.
