# Predictive maintenance copilot

Predicts remaining useful life for turbofan engines from sensor data, turns each
prediction into a grounded maintenance work order, and prices the maintenance policy
against the alternatives.

The point is the end to end system, not the leaderboard: a number like "RUL = 42" is
not actionable on a flight line, and an RMSE does not tell a programme manager whether
to change how the fleet is maintained.

## Run it

```bash
uv sync
uv run python scripts/train.py              # downloads data, trains, writes artifacts
uv run pytest
uv run streamlit run src/copilot/app.py
```

Work order generation calls the Claude API and needs `ANTHROPIC_API_KEY` set.
Everything else runs offline on CPU.

## Results, FD001

| Metric | Result | Target |
|---|---|---|
| Test RMSE | 14.5 cycles | under 15 |
| NASA asymmetric score | 311 | beat the linear baseline (389) |
| Early warning rate | 94% | 90% |
| False alarm rate | 0% | under 10% |
| Interval coverage | 77% | 80% nominal |
| Fleet inference | 9 ms for 100 engines | under 1 s |

Three models, chosen on a validation split of held-out engines. The test set was
scored once, at the end.

| Model | Validation RMSE | Test RMSE | NASA score | Train time |
|---|---|---|---|---|
| Ridge | 18.4 | 16.0 | 389 | 0.01 s |
| **LightGBM** | **15.0** | **14.5** | **311** | 1.7 s |
| 1D-CNN | 16.7 | 15.8 | 405 | 52 s |

Published deep learning results on FD001 sit around 11 to 14 RMSE. The CNN here is
deliberately small and does not reach that, which makes the production choice easy:
the tree model is more accurate, trains in under two seconds, carries its own
uncertainty and attributions, and deploys as a text file.

## How it works

```
C-MAPSS data -> features -> LightGBM (point + quantiles) -> predictions.parquet -> Streamlit
                                                         -> work order (Claude) -> groundedness check
                                                         -> policy simulator
```

`scripts/train.py` does all the computation once and writes artifacts. The app reads
parquet and never loads a model, so the fleet view is instant.

### Modelling

- Labels are piecewise linear, capped at 125 cycles. Degradation is not observable
  before that, so a linear label there would only teach the model to fit noise.
- Features are per engine: normalised sensor values, rolling mean and spread over 5
  and 20 cycles, and a trend term. Seven of the 21 sensors are flat in FD001 and are
  dropped.
- Splits are by engine, never by row. Twenty engines are held out from training for
  validation and conformal calibration. The test set is scored once, at the end.
- Uncertainty comes from quantile regression, calibrated with conformalised quantile
  regression on the held-out engines. Raw quantiles covered 69% of true values against
  a nominal 80%; calibration widens them to 77%.
- Attributions are exact TreeSHAP values from LightGBM's own `pred_contrib`, so there
  is no separate SHAP dependency.

### Work orders

The model receives the prediction, the interval and the attributions as fixed input.
It may explain them and may not change them. Output is constrained to a Pydantic
schema through structured outputs, so schema validity is guaranteed rather than
retried. Every sensor value the work order cites is then checked against the source
data, and a failing work order is flagged in the UI rather than shown as if it were
fine.

### Policy simulator

Three policies over 100 engines: run to failure, fixed interval, and predictive.
Every cost assumption is editable in the sidebar and nothing is hidden in the
arithmetic.

The simulator runs on the **training** engines, using out-of-fold predictions so no
engine is scored by a model that trained on it. This is deliberate. Test trajectories
are truncated at an unknown cycle, usually while the engine is still healthy, so a
policy that acts near end of life cannot be measured on them: 70 of the 100 test
engines never reach a decision point at all, and scoring them makes the predictive
policy look three times worse than a fixed schedule for reasons that have nothing to
do with the model.

The simulator also models parts and slot lead time. Deciding to pull an engine is not
the same as pulling it, and without that delay the model rewards an ever lower
threshold. With it, the cost curve has a genuine optimum around 20 cycles and a cliff
below 15, where warnings arrive too late to act on.

## What is real and what is not

- **Real:** the C-MAPSS dataset, from the NASA Prognostics Center of Excellence. It is
  simulated engine degradation, not measurements from real hardware.
- **Synthetic:** the maintenance logs and the manual extract in `src/copilot/manual.md`.
  Both were written for this project. Part numbers and task codes are invented.
- **Not included:** integration with any real maintenance system, and any airworthiness
  judgement. The system recommends, people decide.

## Layout

| Path | What it does |
|---|---|
| `src/copilot/config.py` | Every threshold, cost assumption, hyperparameter and model id |
| `src/copilot/data.py` | Download, parse, RUL labels, features |
| `src/copilot/models.py` | Training, scoring, intervals, attributions |
| `src/copilot/evaluate.py` | Validation sampling, cross-fitting, replay metrics |
| `src/copilot/workorder.py` | Schema, prompt assembly, LLM call, groundedness check |
| `src/copilot/prompt.md` | The system prompt, versioned on its own |
| `src/copilot/manual.md` | Synthetic maintenance manual extract |
| `src/copilot/telemetry.py` | SQLite record of every LLM call and what it cost |
| `src/copilot/simulator.py` | The three policies and the cost model |
| `src/copilot/charts.py` | Every chart, built from the shared palette |
| `src/copilot/theme.py` | The validated colour palette |
| `src/copilot/app.py` | Four tab Streamlit UI, layout only |
| `scripts/train.py` | Orchestration: the one command that produces everything |
| `tests/test_smoke.py` | Labels, leakage, replay metrics, groundedness, cost model, app render |

Anything that can change a reported number lives in the package and is under test.
`scripts/train.py` only wires those pieces together.

## Known limits

- The 1D-CNN is small and untuned. A deeper model with attention would likely close the
  gap to published results, and is the obvious next modelling step.
- FD001 only: one operating condition, one fault mode. FD002 to FD004 would need per
  condition normalisation.
- Interval coverage is 77% against a nominal 80%. The gap is structural: validation
  labels are capped at 125 cycles while several test engines truly have more life than
  that, so those can never be covered.
- There is no human rating rubric or LLM judge for the work orders yet. The telemetry
  table has room for them.
