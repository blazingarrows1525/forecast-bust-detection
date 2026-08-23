"""Observability: structured logging and Prometheus-format metrics."""
from fbd.obs.metrics import REGISTRY, get_logger, log_event, timed

__all__ = ["REGISTRY", "get_logger", "log_event", "timed"]
