"""How well the models actually do, including where they fail."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from copilot import charts, config, shared


def body(predictions: pd.DataFrame, metrics: dict) -> None:
    st.subheader("Model comparison")
    table = pd.DataFrame(metrics["models"]).T
    st.dataframe(
        table[["val_rmse", "test_rmse", "test_nasa_score", "train_seconds", "fleet_inference_seconds"]]
        .rename(columns={
            "val_rmse": "validation RMSE", "test_rmse": "test RMSE", "test_nasa_score": "NASA score",
            "train_seconds": "train (s)", "fleet_inference_seconds": "fleet inference (s)",
        })
        .round(3)
    )
    st.caption(
        f"Production model chosen on the validation split: **{metrics['production_model']}**. The test set "
        "was scored once, at the end. Published deep-learning results on FD001 sit around 11 to 14 RMSE."
    )

    snapshot = shared.fleet_snapshot(predictions)
    left, right = st.columns(2)
    left.plotly_chart(charts.accuracy_scatter(snapshot), use_container_width=True)
    right.plotly_chart(charts.lead_time_histogram(predictions), use_container_width=True)

    gbm = metrics["models"]["lightgbm"]
    cols = st.columns(3)
    cols[0].metric("Interval coverage", f"{gbm['interval_coverage']:.0%}",
                   help=f"Nominal 80%. Raw quantiles gave {gbm['raw_interval_coverage']:.0%} before "
                        f"conformal calibration widened them by {gbm['conformal_width_cycles']:.1f} cycles")
    cols[1].metric("False alarm rate", f"{gbm['false_alarm_rate']:.0%}",
                   help=f"Share of Red flags raised with more than {config.TIER_AMBER} cycles remaining")
    cols[2].metric("Median lead time", f"{gbm['median_lead_cycles']:.0f} cycles")

    st.subheader("Worst five predictions")
    snapshot = snapshot.assign(error=(snapshot["point"] - snapshot["true_rul"]).abs())
    worst = snapshot.nlargest(5, "error")
    st.dataframe(
        worst[["cycle", "true_rul", "point", "lower", "upper", "error", "driver1", "tier"]]
        .rename(columns={"point": "predicted", "true_rul": "true RUL", "driver1": "top driver"})
        .round(1)
    )
    st.caption(
        "The large errors cluster on engines whose sensors are still near baseline at the truncation "
        "point: with no visible degradation the model falls back to the population average, which is "
        "the capped label of 125 cycles."
    )
