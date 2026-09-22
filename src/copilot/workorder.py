"""Turn a prediction into a maintenance work order, and check that it is grounded.

The model is given the numbers and asked to explain them. It cannot change them: the
RUL, the interval and the attributions are computed upstream and echoed back for
verification. Structured outputs make the schema valid by construction rather than by
retry loop, and every cited sensor value is checked against the data before the work
order is shown.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import anthropic
import pandas as pd
from pydantic import BaseModel, Field

from copilot import config, data, telemetry

_HERE = Path(__file__).parent
MANUAL = (_HERE / "manual.md").read_text()

# The system prompt is a file, not a string constant: prompts change more often than the
# code around them and a separate file keeps those changes reviewable on their own.
SYSTEM_PROMPT = (_HERE / "prompt.md").read_text() + "\n\n" + MANUAL

CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.1


class Interval(BaseModel):
    low: int = Field(description="Lower bound of the predicted RUL interval, in cycles")
    high: int = Field(description="Upper bound of the predicted RUL interval, in cycles")


class Evidence(BaseModel):
    sensor: str = Field(description="Sensor id exactly as given, for example 's11'")
    value: float = Field(description="Current reading, copied verbatim from the supplied table")
    trend: Literal["rising", "falling", "stable"] = Field(
        description="Copy the trend column from the readings table. It is computed, not judged."
    )
    reference_range: str = Field(description="Healthy range for this sensor, from the fleet baseline given")


class WorkOrder(BaseModel):
    """Field order is the order a technician reads them in."""

    engine_id: int
    priority: Literal["Immediate", "Urgent", "Routine", "Monitor"]
    predicted_rul: int
    interval: Interval
    suspected_subsystem: Literal["Fan", "HPC", "HPT", "LPT", "Combustor", "Unknown"]
    evidence: list[Evidence] = Field(min_length=1, max_length=8)
    recommended_actions: list[str] = Field(min_length=1, max_length=6)
    parts_to_stage: list[str] = Field(max_length=6)
    justification: str
    confidence_note: str


@dataclass
class Generated:
    """A work order plus everything needed to audit and price the call."""

    work_order: WorkOrder
    checks: list["Check"]
    latency_s: float
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cost_usd: float
    model: str
    effort: str | None

    @property
    def grounded(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[str]:
        return [f for c in self.checks for f in c.failures]


def maintenance_history(engine_id: int, current_cycle: int, entries: int | None = None) -> list[str]:
    """Synthetic maintenance log for one engine. Seeded, so it is stable across runs.

    Marked synthetic everywhere it is shown. Real history would come from the unit's
    maintenance system, which is deliberately out of scope for this project.
    """
    import random

    entries = config.LOG_ENTRIES_PER_ENGINE if entries is None else entries
    rng = random.Random(config.SEED * 1000 + engine_id)
    tasks = [
        ("TSK-520 gas path performance run", "within acceptance limits"),
        ("TSK-201 borescope, HPC stages 1 to 9", "light leading edge erosion, stage 4, within limits"),
        ("TSK-505 fan blade inspection", "no findings"),
        ("TSK-330 borescope, HPT stage 1", "minor coating loss, monitor"),
        ("TSK-410 LPT borescope", "no findings"),
        ("TSK-214 HPC efficiency trend review", "efficiency down 1.2 percent since overhaul"),
        ("Oil filter change", "no metal in filter"),
        ("TSK-338 HPT clearance and seal check", "clearance at upper limit"),
    ]
    cycles = sorted(rng.sample(range(1, max(current_cycle, 2)), k=min(entries, max(current_cycle - 1, 1))))
    return [f"cycle {c}: {task} - {finding}" for c, (task, finding) in zip(cycles, rng.sample(tasks, len(cycles)))]


def _sensor_table(row: pd.Series, sensors: list[str], history: pd.DataFrame) -> str:
    """Current reading, recent movement and a fleet baseline for each sensor."""
    lines = ["| sensor | description | current | mean over last 20 cycles | trend | healthy range |",
             "|---|---|---|---|---|---|"]
    for s in sensors:
        recent = history[s].tail(config.TREND_WINDOW)
        low, high = healthy_range(history, s)
        trend = observed_trend(history, s, int(row["cycle"]))
        lines.append(
            f"| {s} | {data.SENSOR_DESCRIPTIONS[s]} | {row[s]:.4f} | {recent.mean():.4f} | "
            f"{trend} | {low:.4f} to {high:.4f} |"
        )
    return "\n".join(lines)


def build_prompt(row: pd.Series, sensors: list[str], history: pd.DataFrame) -> str:
    """Assemble the engine context. Everything the model may cite is in here."""
    drivers = [
        f"{row[f'driver{i}']} ({data.SENSOR_DESCRIPTIONS[data.sensor_of(row[f'driver{i}'])]}), "
        f"contribution {row[f'driver{i}_contrib']:+.1f} cycles"
        for i in (1, 2, 3)
        if f"driver{i}" in row.index
    ]
    logs = maintenance_history(int(row["unit"]), int(row["cycle"]))
    return f"""Engine {int(row['unit'])}, currently at cycle {int(row['cycle'])}.

FIXED MODEL OUTPUT, echo exactly:
- predicted remaining useful life: {int(round(row['point']))} cycles
- prediction interval: {int(round(row['lower']))} to {int(round(row['upper']))} cycles ({int(config.TIER_RED)} cycle red threshold)
- risk tier: {row['tier']}

Top model attributions for this prediction:
{chr(10).join('- ' + d for d in drivers)}

Sensor readings at the current cycle. The "fleet healthy range" column is this engine's
own range over its first 20 cycles, when it was healthy:
{_sensor_table(row, sensors, history)}

Maintenance history (synthetic):
{chr(10).join('- ' + entry for entry in logs)}

Write the work order."""


SENSOR_MENTION = re.compile(r"\bs(\d{1,2})\b")
NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
TASK_CODE = re.compile(r"TSK-\d+")
PART_NUMBER = re.compile(r"P-[A-Z]+-\d+")

# The manual is the only vocabulary a work order may draw task codes and parts from.
# Anything else is invented, however plausible it looks.
KNOWN_TASKS = frozenset(TASK_CODE.findall(MANUAL))
KNOWN_PARTS = frozenset(PART_NUMBER.findall(MANUAL))


@dataclass(frozen=True)
class Check:
    """One groundedness rule and what it found. Named so the UI can show the reasoning."""

    name: str
    passed: bool
    detail: str
    failures: tuple[str, ...] = ()


def observed_trend(history: pd.DataFrame, sensor: str, cycle: int) -> str:
    """Which way a sensor is moving over the trend window, from the data alone.

    The test is whether the shift between the start and end of the window is larger
    than the uncertainty in those two five-point averages. Comparing the shift against
    the sensor's own sampling noise is what makes one threshold work for a temperature
    in the hundreds and a bypass ratio near eight; an absolute epsilon, or the width of
    the healthy band, would mean something different for each.
    """
    seg = history.loc[history["cycle"] <= cycle, sensor].tail(config.TREND_WINDOW)
    if len(seg) < 10:
        return "stable"
    head, tail = seg.head(5), seg.tail(5)
    delta = float(tail.mean() - head.mean())
    noise = float(seg.std(ddof=1)) / len(head) ** 0.5
    if noise <= 0 or abs(delta) < config.TREND_SIGMA * noise:
        return "stable"
    return "rising" if delta > 0 else "falling"


def healthy_range(history: pd.DataFrame, sensor: str) -> tuple[float, float]:
    """The baseline handed to the model: this engine's own first 20 cycles."""
    healthy = history[sensor].iloc[:20]
    return float(healthy.min()), float(healthy.max())


def evidence_audit(order: WorkOrder, row: pd.Series, history: pd.DataFrame) -> pd.DataFrame:
    """Every claim in evidence next to what the data says, one row per sensor."""
    records = []
    for item in order.evidence:
        known = item.sensor in row.index
        actual = float(row[item.sensor]) if known else float("nan")
        trend = observed_trend(history, item.sensor, int(row["cycle"])) if known else "unknown"
        low, high = healthy_range(history, item.sensor) if known else (float("nan"), float("nan"))
        cited = [float(n) for n in NUMBER.findall(item.reference_range)]
        range_ok = (
            len(cited) == 2
            and abs(cited[0] - low) <= config.RANGE_TOLERANCE
            and abs(cited[1] - high) <= config.RANGE_TOLERANCE
        )
        records.append({
            "sensor": item.sensor,
            "cited value": item.value,
            "actual": actual,
            "delta": item.value - actual if known else float("nan"),
            "value ok": known and abs(item.value - actual) <= config.SENSOR_TOLERANCE,
            "cited trend": item.trend,
            "observed trend": trend,
            "trend ok": item.trend == trend,
            "cited range": item.reference_range,
            "actual range": f"{low:.4f} to {high:.4f}" if known else "n/a",
            "range ok": range_ok,
        })
    return pd.DataFrame(records)


def check_grounded(order: WorkOrder, row: pd.Series, history: pd.DataFrame) -> list[Check]:
    """Every rule that decides whether a work order can be trusted, with its result.

    Returns one Check per rule rather than a flat list of strings, so the app can show
    what was verified instead of a bare pass or fail. A badge nobody can interrogate is
    not evidence of anything.
    """
    audit = evidence_audit(order, row, history)
    n = len(audit)

    unknown = [s for s in audit.loc[~audit["sensor"].isin(row.index), "sensor"]] if n else []
    values = audit[audit["sensor"].isin(row.index)] if n else audit

    value_failures = tuple(
        f"{r.sensor} cited as {r['cited value']} but the reading is {r.actual:.4f}"
        for _, r in values.iterrows() if not r["value ok"]
    )
    trend_failures = tuple(
        f"{r.sensor} called {r['cited trend']} but the data is {r['observed trend']}"
        for _, r in values.iterrows() if not r["trend ok"]
    )
    range_failures = tuple(
        f"{r.sensor} healthy range cited as {r['cited range']}, actual {r['actual range']}"
        for _, r in values.iterrows() if not r["range ok"]
    )

    cited_sensors = set(audit["sensor"]) if n else set()
    prose = f"{order.justification} {order.confidence_note}"
    mentioned = {f"s{m}" for m in SENSOR_MENTION.findall(prose)}
    orphans = tuple(
        f"{s} is discussed in the justification but has no verified value in evidence"
        for s in sorted(mentioned - cited_sensors, key=lambda x: int(x[1:]))
    )

    instructions = " ".join(order.recommended_actions + order.parts_to_stage + [order.justification])
    cited_tasks = set(TASK_CODE.findall(instructions))
    cited_parts = set(PART_NUMBER.findall(instructions))
    invented = sorted((cited_tasks - KNOWN_TASKS) | (cited_parts - KNOWN_PARTS))
    n_codes = len(cited_tasks) + len(cited_parts)

    id_ok = order.engine_id == int(row["unit"])
    rul_ok = order.predicted_rul == int(round(row["point"]))

    return [
        Check("Sensors exist in the dataset", not unknown,
              f"{n - len(unknown)} of {n} recognised",
              tuple(f"{s} is not a sensor in this dataset" for s in unknown)),
        Check("Cited values match the data", not value_failures,
              f"{len(values) - len(value_failures)} of {len(values)} within \u00b1{config.SENSOR_TOLERANCE}",
              value_failures),
        Check("Trend directions match the data", not trend_failures,
              f"{len(values) - len(trend_failures)} of {len(values)} agree over {config.TREND_WINDOW} cycles",
              trend_failures),
        Check("Healthy ranges match the baseline", not range_failures,
              f"{len(values) - len(range_failures)} of {len(values)} match", range_failures),
        Check("Prose is backed by evidence", not orphans,
              f"{len(mentioned)} sensors named in prose, {len(mentioned) - len(orphans)} verified",
              orphans),
        Check("Task codes and parts exist in the manual", not invented,
              f"{n_codes - len(invented)} of {n_codes} found in the manual",
              tuple(f"{c} does not appear in the maintenance manual" for c in invented)),
        Check("Engine id unchanged", id_ok,
              f"engine {order.engine_id}",
              () if id_ok else (f"cited {order.engine_id}, actual {int(row['unit'])}",)),
        Check("Prediction unchanged", rul_ok,
              f"{order.predicted_rul} cycles",
              () if rul_ok else (f"cited {order.predicted_rul}, model said {int(round(row['point']))}",)),
    ]


def price(model: str, usage) -> float:
    """Cost of one call in dollars, from the published per-million-token rates."""
    if model not in config.LLM_PRICING:
        raise KeyError(f"no pricing configured for {model}")
    input_rate, output_rate = config.LLM_PRICING[model]
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    return (
        usage.input_tokens * input_rate
        + cache_read * input_rate * CACHE_READ_MULTIPLIER
        + cache_write * input_rate * CACHE_WRITE_MULTIPLIER
        + usage.output_tokens * output_rate
    ) / 1_000_000


def generate(
    row: pd.Series,
    sensors: list[str],
    history: pd.DataFrame,
    model: str | None = None,
    effort: str | None = None,
) -> Generated:
    """Call the model, validate the result and record what it cost."""
    model = model or config.LLM_MODEL
    effort = effort or config.LLM_EFFORT
    client = anthropic.Anthropic()
    # Models that do not support effort reject the parameter outright.
    extra = {"output_config": {"effort": effort}} if model in config.LLM_EFFORT_MODELS else {}

    start = time.perf_counter()
    response = client.messages.parse(
        model=model,
        max_tokens=config.LLM_MAX_TOKENS,
        # The system prompt carries the manual and never changes, so it caches well.
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": build_prompt(row, sensors, history)}],
        output_format=WorkOrder,
        **extra,
    )
    latency = time.perf_counter() - start

    order = response.parsed_output
    if order is None:
        raise RuntimeError(f"model returned no parsed work order, stop reason {response.stop_reason}")

    generated = Generated(
        work_order=order,
        checks=check_grounded(order, row, history),
        latency_s=latency,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        cost_usd=price(model, response.usage),
        model=model,
        effort=effort if extra else None,
    )
    telemetry.record(
        engine_id=int(row["unit"]),
        cycle=int(row["cycle"]),
        model=model,
        effort=generated.effort,
        latency_s=latency,
        input_tokens=generated.input_tokens,
        output_tokens=generated.output_tokens,
        cache_read_tokens=generated.cache_read_tokens,
        cost_usd=generated.cost_usd,
        grounded=generated.grounded,
        failures=generated.failures,
        payload=order.model_dump_json(),
    )
    return generated
