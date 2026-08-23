"""Observability: structured logs and Prometheus-format metrics.

No prometheus_client dependency.  The exposition format is a documented text
protocol and emitting it directly keeps the serving image small and the
air-gapped path free of a library that wants to open a port of its own.  If
this ever outgrows a single node, swap the registry -- the call sites do not
change.

What is measured is chosen to answer operational questions a duty forecaster or
an on-call engineer would actually ask: is the API up, is it slow, is the model
refusing more than usual, and is the GenAI layer being blocked by guardrails.
That last one is a safety signal, not a vanity metric -- a rising
``fbd_guardrail_violations_total`` means something upstream is trying to put
ungrounded numbers in front of a forecaster.
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field

_LOCK = threading.Lock()

# Latency buckets in seconds. Tuned to the operational window: the whole
# bulletin must render inside a 45-minute shift, but a forecaster clicking a
# region expects sub-second. Anything past 5s is a failure worth alerting on.
_BUCKETS = (0.005, 0.025, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)


@dataclass
class _Histogram:
    buckets: dict[float, int] = field(default_factory=lambda: {b: 0 for b in _BUCKETS})
    count: int = 0
    total: float = 0.0

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value
        for edge in _BUCKETS:
            if value <= edge:
                self.buckets[edge] += 1


class Registry:
    """Minimal counter/gauge/histogram registry."""

    def __init__(self) -> None:
        self._counters: dict[tuple[str, tuple], float] = defaultdict(float)
        self._gauges: dict[tuple[str, tuple], float] = {}
        self._histograms: dict[tuple[str, tuple], _Histogram] = defaultdict(_Histogram)
        self._help: dict[str, str] = {}

    @staticmethod
    def _key(name: str, labels: dict | None) -> tuple[str, tuple]:
        return name, tuple(sorted((labels or {}).items()))

    def counter(self, name: str, help_text: str = "", labels: dict | None = None, by: float = 1.0) -> None:
        with _LOCK:
            self._help.setdefault(name, help_text)
            self._counters[self._key(name, labels)] += by

    def gauge(self, name: str, value: float, help_text: str = "", labels: dict | None = None) -> None:
        with _LOCK:
            self._help.setdefault(name, help_text)
            self._gauges[self._key(name, labels)] = value

    def observe(self, name: str, value: float, help_text: str = "", labels: dict | None = None) -> None:
        with _LOCK:
            self._help.setdefault(name, help_text)
            self._histograms[self._key(name, labels)].observe(value)

    @staticmethod
    def _fmt_labels(labels: tuple) -> str:
        if not labels:
            return ""
        inner = ",".join(f'{k}="{v}"' for k, v in labels)
        return "{" + inner + "}"

    def render(self) -> str:
        """Prometheus text exposition format."""
        lines: list[str] = []
        with _LOCK:
            emitted: set[str] = set()

            def header(name: str, kind: str) -> None:
                if name in emitted:
                    return
                emitted.add(name)
                if self._help.get(name):
                    lines.append(f"# HELP {name} {self._help[name]}")
                lines.append(f"# TYPE {name} {kind}")

            for (name, labels), value in sorted(self._counters.items()):
                header(name, "counter")
                lines.append(f"{name}{self._fmt_labels(labels)} {value}")
            for (name, labels), value in sorted(self._gauges.items()):
                header(name, "gauge")
                lines.append(f"{name}{self._fmt_labels(labels)} {value}")
            for (name, labels), hist in sorted(self._histograms.items()):
                header(name, "histogram")
                base = self._fmt_labels(labels)[:-1] if labels else "{"
                sep = "," if labels else ""
                for edge in _BUCKETS:
                    lines.append(f'{name}_bucket{base}{sep}le="{edge}"}} {hist.buckets[edge]}')
                lines.append(f'{name}_bucket{base}{sep}le="+Inf"}} {hist.count}')
                lines.append(f"{name}_sum{self._fmt_labels(labels)} {hist.total}")
                lines.append(f"{name}_count{self._fmt_labels(labels)} {hist.count}")
        return "\n".join(lines) + "\n"


REGISTRY = Registry()


@contextmanager
def timed(name: str, help_text: str = "", labels: dict | None = None):
    """Time a block and record it as a histogram observation."""
    start = time.perf_counter()
    try:
        yield
    finally:
        REGISTRY.observe(name, time.perf_counter() - start, help_text, labels)


# --------------------------------------------------------------------------
# Structured logging
# --------------------------------------------------------------------------
class JsonFormatter(logging.Formatter):
    """One JSON object per line, so logs are queryable without a parser."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in getattr(record, "extra_fields", {}).items():
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str = "fbd") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def log_event(logger: logging.Logger, msg: str, **fields) -> None:
    """Log with structured fields attached."""
    logger.info(msg, extra={"extra_fields": fields})
