"""One command: raw data in, trained models and prediction artifacts out.

    uv run python scripts/train.py

Writes artifacts/predictions.parquet (every test engine at every observed cycle),
artifacts/metrics.json and artifacts/models/. The Streamlit app reads those files
and never calls a model, which is what keeps the fleet view instant.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from copilot import config, data, evaluate, models


def main() -> None:
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    data.download()

    train_raw = data.load("train")
    test_raw = data.load("test")
    true_rul = data.load_true_rul()

    fit_units, val_units = data.split_engines(train_raw)
    fit_raw = train_raw[train_raw["unit"].isin(fit_units)]

    sensors = data.useful_sensors(fit_raw)
    stats = data.fit_stats(fit_raw, sensors)
    print(f"{len(sensors)} usable sensors, {len(fit_units)} fit engines, {len(val_units)} validation engines")

    train_labelled = data.add_rul(train_raw)
    test_labelled = data.add_rul(test_raw, true_rul)

    train_x = data.featurize(train_raw, sensors, stats)
    test_x = data.featurize(test_raw, sensors, stats)
    features = data.feature_columns(train_x)

    is_fit = train_raw["unit"].isin(fit_units).to_numpy()
    x_fit, y_fit = train_x.loc[is_fit, features], train_labelled.loc[is_fit, "rul_capped"]
    val_idx = evaluate.validation_points(train_raw.loc[~is_fit])
    x_val, y_val = train_x.loc[val_idx, features], train_labelled.loc[val_idx, "rul_capped"]

    # Test evaluation happens once, on the final observed cycle of each engine.
    test_last = test_labelled.groupby("unit")["cycle"].idxmax()
    x_test = test_x.loc[test_last, features]
    y_test = test_labelled.loc[test_last, "rul"].to_numpy()

    results: dict[str, dict] = {}

    t0 = time.perf_counter()
    ridge = models.train_ridge(x_fit, y_fit)
    ridge_train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    ridge_test = np.clip(ridge.predict(x_test), 0, config.RUL_CAP)
    results["ridge"] = {
        "val_rmse": models.rmse(y_val, np.clip(ridge.predict(x_val), 0, config.RUL_CAP)),
        "test_rmse": models.rmse(y_test, ridge_test),
        "test_nasa_score": models.nasa_score(y_test, ridge_test),
        "train_seconds": ridge_train_s,
        "fleet_inference_seconds": time.perf_counter() - t0,
    }

    t0 = time.perf_counter()
    gbm = models.train_gbm(x_fit, y_fit, x_val, y_val)
    gbm_train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    gbm_test = models.predict_gbm(gbm, x_test)
    gbm_val = models.predict_gbm(gbm, x_val)
    width = models.conformal_width(y_val.to_numpy(), gbm_val)
    gbm_test = models.apply_conformal(gbm_test, width)
    results["lightgbm"] = {
        "val_rmse": models.rmse(y_val, gbm_val["point"]),
        "test_rmse": models.rmse(y_test, gbm_test["point"]),
        "test_nasa_score": models.nasa_score(y_test, gbm_test["point"]),
        "train_seconds": gbm_train_s,
        "fleet_inference_seconds": time.perf_counter() - t0,
        "raw_interval_coverage": float(((y_test >= gbm_test["q10"]) & (y_test <= gbm_test["q90"])).mean()),
        "interval_coverage": float(((y_test >= gbm_test["lower"]) & (y_test <= gbm_test["upper"])).mean()),
        "conformal_width_cycles": width,
        "best_iterations": {name: m.best_iteration for name, m in gbm.items()},
    }

    # 1D-CNN over 30-cycle sensor windows, for the comparison in the Evaluation tab.
    windows_train, window_index = models.make_windows(train_x, sensors)
    position = pd.Series(np.arange(len(window_index)), index=window_index)
    t0 = time.perf_counter()
    cnn = models.train_cnn(
        windows_train[position.loc[train_x.index[is_fit]].to_numpy()],
        y_fit.to_numpy(),
        windows_train[position.loc[val_idx].to_numpy()],
        y_val.to_numpy(),
    )
    cnn_train_s = time.perf_counter() - t0

    windows_test, test_window_index = models.make_windows(test_x, sensors)
    test_position = pd.Series(np.arange(len(test_window_index)), index=test_window_index)
    t0 = time.perf_counter()
    cnn_test = models.predict_cnn(cnn, windows_test[test_position.loc[test_last.to_numpy()].to_numpy()])
    results["cnn"] = {
        "val_rmse": models.rmse(
            y_val, models.predict_cnn(cnn, windows_train[position.loc[val_idx].to_numpy()])
        ),
        "test_rmse": models.rmse(y_test, cnn_test),
        "test_nasa_score": models.nasa_score(y_test, cnn_test),
        "train_seconds": cnn_train_s,
        "fleet_inference_seconds": time.perf_counter() - t0,
    }

    production = min(results, key=lambda name: results[name]["val_rmse"])
    print(f"production model chosen on validation: {production}")
    for name, r in results.items():
        print(f"  {name:9s} val RMSE {r['val_rmse']:6.2f}   test RMSE {r['test_rmse']:6.2f}   NASA {r['test_nasa_score']:9.1f}")

    # Full cycle-by-cycle predictions for the fleet view, replay and the simulator.
    preds = models.apply_conformal(models.predict_gbm(gbm, test_x[features]), width)
    drivers = models.top_attributions(gbm["point"], test_x[features])
    frame = pd.concat(
        [
            test_labelled[["unit", "cycle", "rul"]].rename(columns={"rul": "true_rul"}),
            test_labelled[sensors],
            preds,
            drivers,
        ],
        axis=1,
    )
    frame["tier"] = models.tier(frame["point"])
    frame["last_cycle"] = frame.groupby("unit")["cycle"].transform("max")
    frame["failure_cycle"] = frame["cycle"] + frame["true_rul"]
    frame.to_parquet(config.PREDICTIONS, index=False)

    rounds = {name: m.best_iteration for name, m in gbm.items()}
    simulation = evaluate.out_of_fold(train_raw, train_labelled, train_x, features, sensors, rounds, width)
    simulation.to_parquet(config.SIMULATION, index=False)

    results["lightgbm"].update(evaluate.replay_metrics(frame))
    metrics = {
        "dataset": config.DATASET,
        "rul_cap": config.RUL_CAP,
        "sensors_used": sensors,
        "n_test_engines": int(frame["unit"].nunique()),
        "n_simulation_engines": int(simulation["unit"].nunique()),
        "production_model": production,
        "models": results,
    }
    config.METRICS.write_text(json.dumps(metrics, indent=2))

    for name, model in gbm.items():
        model.save_model(str(config.MODEL_DIR / f"gbm_{name}.txt"))
    torch.save(cnn.state_dict(), config.MODEL_DIR / "cnn.pt")
    stats.to_csv(config.MODEL_DIR / "normalisation.csv")
    (config.MODEL_DIR / "conformal_width.txt").write_text(str(width))

    early = results["lightgbm"]["early_warning_rate"]
    print(f"interval coverage {results['lightgbm']['interval_coverage']:.0%}, early warning {early:.0%}, "
          f"false alarms {results['lightgbm']['false_alarm_rate']:.0%}")
    print(f"wrote {config.PREDICTIONS.name} ({len(frame)} rows), "
          f"{config.SIMULATION.name} ({len(simulation)} rows) and {config.METRICS.name}")


if __name__ == "__main__":
    main()
