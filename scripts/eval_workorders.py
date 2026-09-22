"""Run the work order generator over a sample of engines and measure what comes back.

    uv run python scripts/eval_workorders.py --engines 6 --repeats 2

Answers two questions that single calls cannot. How often does a work order pass every
groundedness rule, and which rule fails when one does. And when the same engine is
described twice, does the model reach the same conclusion, which matters because
`suspected_subsystem` decides which module gets opened.

This costs real money. The estimate is printed before anything is sent.
"""

from __future__ import annotations

import argparse
import collections
import json
import time

import pandas as pd

from copilot import config, telemetry, workorder


def sample_engines(predictions: pd.DataFrame, n: int) -> list[int]:
    """A spread across risk tiers, so the sample is not all end-of-life engines."""
    snapshot = predictions.loc[predictions.groupby("unit")["cycle"].idxmax()]
    picks: list[int] = []
    for tier in ("Red", "Amber", "Green"):
        group = snapshot[snapshot["tier"] == tier].sort_values("point")
        take = max(1, round(n * len(group) / len(snapshot)))
        picks += [int(u) for u in group["unit"].head(take)]
    return picks[:n]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engines", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=2, help="runs per engine, to measure agreement")
    parser.add_argument("--models", nargs="+", default=["claude-opus-5", "claude-haiku-4-5"])
    parser.add_argument("--effort", default=config.LLM_EFFORT)
    args = parser.parse_args()

    predictions = pd.read_parquet(config.PREDICTIONS)
    sensors = json.loads(config.METRICS.read_text())["sensors_used"]
    engines = sample_engines(predictions, args.engines)
    calls = len(engines) * args.repeats * len(args.models)

    # Rough per-call costs measured on this prompt, only to set expectations.
    rough = {"claude-opus-5": 0.06, "claude-sonnet-5": 0.025, "claude-haiku-4-5": 0.008}
    estimate = sum(rough.get(m, 0.06) for m in args.models) * len(engines) * args.repeats
    print(f"{calls} calls over engines {engines}, estimated ${estimate:.2f}")

    records = []
    start = time.perf_counter()
    for model in args.models:
        for unit in engines:
            history = predictions[predictions["unit"] == unit].reset_index(drop=True)
            row = history.iloc[-1]
            for attempt in range(args.repeats):
                try:
                    generated = workorder.generate(row, sensors, history, model=model, effort=args.effort)
                except Exception as error:  # a failed call is a result, not a reason to stop
                    print(f"  {model} engine {unit} run {attempt}: {type(error).__name__}")
                    records.append({"model": model, "unit": unit, "error": type(error).__name__})
                    continue
                order = generated.work_order
                records.append({
                    "model": model,
                    "unit": unit,
                    "tier": row["tier"],
                    "grounded": generated.grounded,
                    "failed_rules": [c.name for c in generated.checks if not c.passed],
                    "subsystem": order.suspected_subsystem,
                    "priority": order.priority,
                    "evidence": len(order.evidence),
                    "latency_s": generated.latency_s,
                    "cost_usd": generated.cost_usd,
                })
                print(f"  {model} engine {unit} run {attempt}: "
                      f"{'pass' if generated.grounded else 'FAIL'} {order.suspected_subsystem}")

    frame = pd.DataFrame(records)
    report = summarise(frame)
    report["wall_seconds"] = round(time.perf_counter() - start, 1)
    report["spend_usd"] = round(float(frame.get("cost_usd", pd.Series(dtype=float)).sum()), 4)
    (config.ARTIFACT_DIR / "workorder_eval.json").write_text(json.dumps(report, indent=2))

    print(f"\nspent ${report['spend_usd']:.2f} in {report['wall_seconds']:.0f}s")
    print(json.dumps({k: v for k, v in report.items() if k != "per_model"}, indent=2))
    print(pd.DataFrame(report["per_model"]).T.to_string())


def summarise(frame: pd.DataFrame) -> dict:
    """Groundedness rate, which rules fail, and whether repeats agree."""
    ok = frame[frame.get("grounded").notna()] if "grounded" in frame else frame.iloc[0:0]
    per_model = {}
    for model, group in ok.groupby("model"):
        agreement = subsystem_agreement(group)
        per_model[model] = {
            "calls": len(group),
            "groundedness_rate": round(float(group["grounded"].mean()), 3),
            "within_model_agreement": agreement,
            "mean_latency_s": round(float(group["latency_s"].mean()), 1),
            "mean_cost_usd": round(float(group["cost_usd"].mean()), 4),
            "mean_evidence": round(float(group["evidence"].mean()), 1),
        }

    rules = collections.Counter(r for rs in ok.get("failed_rules", []) for r in rs)
    return {
        "calls": int(len(frame)),
        "errors": int(frame["error"].notna().sum()) if "error" in frame else 0,
        "groundedness_rate": round(float(ok["grounded"].mean()), 3) if len(ok) else 0.0,
        "failed_rules": dict(rules),
        # Pooled across models, so this answers "would two models open the same module",
        # while the per-model figure answers "is one model self-consistent".
        "cross_model_agreement": subsystem_agreement(ok),
        "per_model": per_model,
    }


def subsystem_agreement(frame: pd.DataFrame) -> float:
    """Share of engines where every repeat named the same subsystem.

    Engines seen only once are excluded: one observation cannot agree or disagree.
    """
    repeated = [g for _, g in frame.groupby("unit") if len(g) > 1]
    if not repeated:
        return float("nan")
    return round(sum(g["subsystem"].nunique() == 1 for g in repeated) / len(repeated), 3)


if __name__ == "__main__":
    main()
