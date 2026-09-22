"""Labels and splits. A quiet mistake here poisons every number downstream."""

from __future__ import annotations

import numpy as np
import pandas as pd

from copilot import config, data


def test_rul_labels_are_piecewise_capped():
    """A run-to-failure engine counts down to zero, flat above the cap."""
    frame = pd.DataFrame({"unit": [1] * 5, "cycle": [1, 2, 3, 4, 5]})
    labelled = data.add_rul(frame)
    assert labelled["rul"].tolist() == [4, 3, 2, 1, 0]
    assert labelled["rul_capped"].tolist() == [4, 3, 2, 1, 0]

    long_engine = pd.DataFrame({"unit": [1] * 200, "cycle": range(1, 201)})
    capped = data.add_rul(long_engine)["rul_capped"]
    assert capped.max() == config.RUL_CAP
    assert capped.iloc[-1] == 0
    assert (capped.iloc[:75] == config.RUL_CAP).all()


def test_test_engine_rul_is_offset_by_the_true_remaining_life():
    """Test trajectories stop early, so their labels have to carry the offset."""
    frame = pd.DataFrame({"unit": [7] * 3, "cycle": [1, 2, 3]})
    labelled = data.add_rul(frame, pd.Series({7: 50}))
    assert labelled["rul"].tolist() == [52, 51, 50]


def test_split_is_by_engine_not_by_row():
    """Splitting rows instead of engines would leak a failure trajectory across the split."""
    train = pd.DataFrame({"unit": np.repeat(np.arange(1, 101), 3), "cycle": [1, 2, 3] * 100})
    fit, val = data.split_engines(train)
    assert len(val) == config.VAL_ENGINES
    assert len(fit) + len(val) == 100
    assert not set(fit) & set(val)
