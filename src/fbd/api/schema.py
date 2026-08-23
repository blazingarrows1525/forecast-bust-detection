"""Output schema.  Never a bare number (LOGIC.md sec 9).

Every field exists because omitting it would let the system be confidently
wrong.  `data_quality` and `input_age_hours` make staleness visible;
`confidence_in_estimate` and `prediction_interval` say how much to trust the
probability itself; `dominant_factors` carries the mandated explanation; and
`status` can say OOD/UNAVAILABLE instead of inventing a number.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class DataQuality(str, Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"   # inputs incomplete but usable
    STALE = "STALE"         # inputs older than the staleness threshold
    UNAVAILABLE = "UNAVAILABLE"


class PredictionStatus(str, Enum):
    OK = "OK"
    # The state is unlike anything in training.  We refuse to output a number
    # rather than extrapolate (LOGIC.md sec 13).
    OUT_OF_DISTRIBUTION = "OUT_OF_DISTRIBUTION"
    # Fell back to climatological bust rate because the model could not run.
    CLIMATOLOGY_FALLBACK = "CLIMATOLOGY_FALLBACK"
    UNAVAILABLE = "UNAVAILABLE"


class ReviewTier(str, Enum):
    """Operational tier. Three, not two, because REFUSE is not "safe".

    Refused region-days bust ~7x more often than accepted ones (LOGIC.md 11.2),
    so collapsing REFUSE into "not flagged" would hide the highest-risk days.
    """
    AUTO_OK = "AUTO_OK"
    REVIEW = "REVIEW"
    REFUSE = "REFUSE"


class RegimeVector(BaseModel):
    active_monsoon: float = 0.0
    break_monsoon: float = 0.0
    monsoon_depression: float = 0.0
    western_disturbance: float = 0.0
    orographic: float = 0.0
    coastal: float = 0.0
    entropy: float = Field(
        0.0, description="Normalised entropy; high means the regime is ambiguous"
    )


class BustPrediction(BaseModel):
    region: str
    region_id: str
    lead_day: int = Field(ge=1, le=10)
    init_date: str
    valid_date: str

    status: PredictionStatus = PredictionStatus.OK
    bust_probability: float | None = Field(
        None, ge=0.0, le=1.0,
        description="None when status is not OK -- the system declines to guess",
    )
    confidence_in_estimate: float | None = Field(None, ge=0.0, le=1.0)
    prediction_interval: list[float] | None = None

    dominant_factors: list[str] = Field(default_factory=list)
    regime: RegimeVector | None = None

    review_tier: ReviewTier = Field(
        ReviewTier.AUTO_OK,
        description="AUTO_OK / REVIEW / REFUSE. REFUSE means unknown risk, not low risk.",
    )
    tier_guidance: str | None = Field(
        None, description="What the tier means for the duty forecaster, in plain words."
    )

    data_quality: DataQuality = DataQuality.OK
    input_age_hours: float | None = None
    ood_distance: float | None = None

    # Calibrated ensemble-spread baseline for the same row, so the UI can toggle
    # model vs baseline on the same map and a judge can see the gap directly.
    baseline_probability: float | None = None

    # Kept for the verification log; absent for a live forecast.
    observed_rain_mm: float | None = None
    forecast_rain_mm: float | None = None
    actual_bust: int | None = None

    model_version: str = "0.1.0"


class Bulletin(BaseModel):
    init_date: str
    issued_at: str
    lead_day: int | None = None
    n_regions: int
    data_quality: DataQuality
    input_age_hours: float | None = None
    banner: str | None = Field(
        None, description="Human-readable warning shown across the top of the map"
    )
    predictions: list[BustPrediction]


class ReviewQueueItem(BaseModel):
    rank: int
    region: str
    region_id: str
    lead_day: int
    valid_date: str
    bust_probability: float | None
    status: PredictionStatus
    forecast_rain_mm: float | None
    dominant_factors: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    model_version: str
    model_loaded: bool
    n_bulletin_rows: int
    init_date_range: list[str] | None
    latest_init_date: str | None
    input_age_hours: float | None
    data_quality: DataQuality
    drift_status: dict | None = Field(
        None,
        description="KS drift of recent inputs vs the training snapshot: OK/WATCH/DRIFT/UNKNOWN.",
    )
    notes: list[str] = Field(default_factory=list)


class OverrideRequest(BaseModel):
    """A forecaster dismissing or escalating a flag.  Always audited.

    The system never issues or suppresses a public warning; a human forecaster is
    the authority.  Overrides are stored with the user and the reason so that any
    decision can be reconstructed after an event, and they become training data.
    """
    region_id: str
    init_date: str
    lead_day: int
    action: Literal["dismiss", "escalate"]
    reason: str = Field(min_length=3, max_length=500)
    user: str = Field(min_length=1, max_length=100)
