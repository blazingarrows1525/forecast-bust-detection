"""Shared fixtures.

The one thing here worth explaining is ``bulletin_store``.

``data/artifacts/bulletins.sqlite`` is ~63 MB and gitignored (DATA.md: it ships
as a checksum-verified Release asset, it is not in git history).  Tests that
asserted on what a tool *returned* therefore passed on a developer machine that
had generated it and failed in a fresh checkout -- which is how CI ended up red
on ``tools must contribute grounded numbers`` while the same test was green
locally.  A test whose result depends on whether an untracked 63 MB file
happens to exist is not testing the thing it claims to test.

So the fixture builds a three-row store in a tmp dir and points
``fbd.genai.tools.DB`` at it.  The agent-loop invariant (tool results must flow
into the grounded-number set the guardrail checks against) is then asserted
against data the test itself controls, in any checkout, with no downloads.
"""
from __future__ import annotations

import sqlite3

import pytest

# Mirrors the production schema in scripts/generate_bulletins.py.  Only the
# columns the read-only tools actually SELECT are populated; the rest are left
# NULL so a column the tools start depending on fails loudly here too.
_SCHEMA = """
CREATE TABLE bulletins (
    region_id TEXT NOT NULL,
    region TEXT NOT NULL,
    init_date TEXT NOT NULL,
    lead_day INTEGER NOT NULL,
    valid_date TEXT NOT NULL,
    status TEXT NOT NULL,
    bust_probability REAL,
    confidence_in_estimate REAL,
    pi_low REAL,
    pi_high REAL,
    dominant_factors TEXT,
    regime_json TEXT,
    data_quality TEXT NOT NULL,
    ood_distance REAL,
    forecast_rain_mm REAL,
    observed_rain_mm REAL,
    actual_bust INTEGER,
    baseline_probability REAL,
    model_version TEXT NOT NULL,
    PRIMARY KEY (region_id, init_date, lead_day)
);
"""

# Shapes match a real bulletin: one high-risk accepted row, one mid, and one
# OOD refusal carrying no probability -- the three states the assistant has to
# describe without inventing anything.
_ROWS = [
    ("ASSAM_MEGHALAYA", "Assam & Meghalaya", "2022-06-14", 3, "2022-06-17",
     "OK", 0.706, 0.62, 18.4, 74.1, '["forecast rain +61.9 mm above normal"]',
     None, "OK", 2.31, 88.2, None, None, 0.041, "test-fixture"),
    ("KONKAN_GOA", "Konkan & Goa", "2022-06-14", 4, "2022-06-18",
     "OK", 0.412, 0.55, 9.7, 51.3, '["ensemble spread 2.4x climatology"]',
     None, "OK", 1.88, 46.0, None, None, 0.041, "test-fixture"),
    ("WEST_RAJASTHAN", "West Rajasthan", "2022-06-14", 5, "2022-06-19",
     "OUT_OF_DISTRIBUTION", None, None, None, None, "[]",
     None, "OK", 41.7, 3.2, None, None, 0.041, "test-fixture"),
]


@pytest.fixture
def bulletin_store(tmp_path, monkeypatch):
    """A minimal bulletins.sqlite, with the read-only tools pointed at it."""
    from fbd.genai import tools

    db = tmp_path / "bulletins.sqlite"
    con = sqlite3.connect(db)
    con.executescript(_SCHEMA)
    con.executemany(
        f"INSERT INTO bulletins VALUES ({','.join('?' * len(_ROWS[0]))})", _ROWS
    )
    con.commit()
    con.close()

    monkeypatch.setattr(tools, "DB", db)
    return db
