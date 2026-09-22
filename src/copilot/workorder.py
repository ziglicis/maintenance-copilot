"""Turn a prediction into a maintenance work order, and check that it is grounded.

The model is given the numbers and asked to explain them. It cannot change them: the
RUL, the interval and the attributions are computed upstream and echoed back for
verification. Structured outputs make the schema valid by construction rather than by
retry loop, and every cited sensor value is checked against the data before the work
order is shown.
"""

from __future__ import annotations

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
    trend: Literal["rising", "falling", "stable"]
    reference_range: str = Field(description="Healthy range for this sensor, from the fleet baseline given")


class WorkOrder(BaseModel):
    """Field order is the order a technician reads them in."""

    engine_id: int
    priority: Literal["Immediate", "Urgent", "Routine", "Monitor"]
    predicted_rul: int
    interval: Interval
    suspected_subsystem: Literal["Fan", "HPC", "HPT", "LPT", "Combustor", "Unknown"]
    evidence: list[Evidence] = Field(min_length=1, max_length=6)
    recommended_actions: list[str] = Field(min_length=1, max_length=6)
    parts_to_stage: list[str] = Field(max_length=6)
    justification: str
    confidence_note: str


@dataclass
class Generated:
    """A work order plus everything needed to audit and price the call."""

    work_order: WorkOrder
    grounded: bool
    failures: list[str]
    latency_s: float
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cost_usd: float
    model: str


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
    lines = ["| sensor | description | current | mean over last 20 cycles | fleet healthy range |",
             "|---|---|---|---|---|"]
    for s in sensors:
        recent = history[s].tail(20)
        baseline = f"{history[s].iloc[:20].min():.4f} to {history[s].iloc[:20].max():.4f}"
        lines.append(
            f"| {s} | {data.SENSOR_DESCRIPTIONS[s]} | {row[s]:.4f} | {recent.mean():.4f} | {baseline} |"
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


def check_grounded(order: WorkOrder, row: pd.Series) -> list[str]:
    """Verify every cited sensor value against the data.

    Returns a list of problems, empty when the work order is fully grounded. The check
    is deliberately strict: a plausible but wrong number is the failure mode that makes
    a system like this untrustworthy, so a near miss is still a miss.
    """
    failures = []
    for item in order.evidence:
        if item.sensor not in row.index:
            failures.append(f"{item.sensor} is not a sensor in this dataset")
            continue
        actual = float(row[item.sensor])
        if abs(item.value - actual) > config.SENSOR_TOLERANCE:
            failures.append(f"{item.sensor} cited as {item.value} but the reading is {actual:.4f}")
    if order.engine_id != int(row["unit"]):
        failures.append(f"engine id cited as {order.engine_id} but this is engine {int(row['unit'])}")
    if order.predicted_rul != int(round(row["point"])):
        failures.append(f"RUL cited as {order.predicted_rul} but the model predicted {int(round(row['point']))}")
    return failures


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


def generate(row: pd.Series, sensors: list[str], history: pd.DataFrame, model: str | None = None) -> Generated:
    """Call the model, validate the result and record what it cost."""
    model = model or config.LLM_MODEL
    client = anthropic.Anthropic()

    start = time.perf_counter()
    response = client.messages.parse(
        model=model,
        max_tokens=config.LLM_MAX_TOKENS,
        # The system prompt carries the manual and never changes, so it caches well.
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": build_prompt(row, sensors, history)}],
        output_format=WorkOrder,
    )
    latency = time.perf_counter() - start

    order = response.parsed_output
    if order is None:
        raise RuntimeError(f"model returned no parsed work order, stop reason {response.stop_reason}")

    failures = check_grounded(order, row)
    generated = Generated(
        work_order=order,
        grounded=not failures,
        failures=failures,
        latency_s=latency,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
        cost_usd=price(model, response.usage),
        model=model,
    )
    telemetry.record(
        engine_id=int(row["unit"]),
        cycle=int(row["cycle"]),
        model=model,
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
