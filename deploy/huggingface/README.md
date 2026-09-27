---
title: Forecast Bust Detection
emoji: 🌧️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8912
pinned: false
short_description: Flags when a rainfall forecast over India is about to fail
---

# Forecast Bust Detection

This predicts **when an existing medium-range rainfall forecast over India is
about to fail**: which subdivision, which lead day, and why. It does not
predict the weather.

The number on the map is the probability that a forecast busts. It combines
a calibrated XGBoost model with the spread of the 50-member IFS ensemble.
Every flag carries plain-language meteorological reasons. Days outside the
model's experience are refused rather than guessed at.

- **Dashboard:** `/` (map and review queue), `/command.html` (3-D risk cube),
  `/landing.html` (the evidence)
- **API:** `/docs`
- **Data:** the 2021–2022 monsoon seasons, precomputed. Nothing runs live.

This Space is a read-only demo: the forecaster override log is disabled here.
The source, the method, every registered result and the full data policy are
in the GitHub repository: <https://github.com/blazingarrows1525/forecast-bust-detection>.

Decision support only. It never issues or suppresses a public warning; a human
forecaster is always the authority.
