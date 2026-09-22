"""Every page renders. Catches a broken chart or a renamed column that data tests miss."""

from __future__ import annotations

import pytest

from copilot import config


def _render_view(name: str) -> None:
    """Render one view with real artifacts, the way the entrypoint wires it up."""
    # Imported inside: AppTest.from_function runs this source in a fresh namespace,
    # so nothing imported at module level in this test file is visible here.
    from copilot import config, shared, simulator
    from copilot.views import business_case, engine, evaluation, fleet, operations, overview

    predictions = shared.load_predictions()
    simulation = shared.load_simulation()
    metrics = shared.load_metrics()
    model = config.LLM_MODEL

    {
        "overview": lambda: overview.body(metrics, simulation, simulator.Assumptions()),
        "fleet": lambda: fleet.body(predictions),
        "engine": lambda: engine.body(predictions, metrics, model),
        "business_case": lambda: business_case.body(simulation),
        "evaluation": lambda: evaluation.body(predictions, metrics),
        "operations": lambda: operations.body(metrics, model),
    }[name]()


@pytest.mark.skipif(not config.PREDICTIONS.exists(), reason="run scripts/train.py first")
@pytest.mark.parametrize("view", ["overview", "fleet", "engine", "business_case", "evaluation", "operations"])
def test_each_view_renders_without_error(view):
    """Catches a broken chart or a renamed column, which data tests will not."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(_render_view, kwargs={"name": view}, default_timeout=120)
    app.run()
    assert not app.exception, app.exception


@pytest.mark.skipif(not config.PREDICTIONS.exists(), reason="run scripts/train.py first")
def test_the_router_builds_every_page():
    """The entrypoint itself: navigation config, sidebar widgets and the default page."""
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(config.ROOT / "src" / "copilot" / "app.py"), default_timeout=120)
    app.run()
    assert not app.exception, app.exception
