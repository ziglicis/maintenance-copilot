"""RUL models, scoring functions and per-prediction attributions.

Three models are compared: a Ridge floor, LightGBM quantile regression (the
production candidate), and a 1D-CNN. LightGBM carries the uncertainty and the
attributions, so no separate ensemble or SHAP dependency is needed:
`predict(pred_contrib=True)` returns exact TreeSHAP values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge

import lightgbm as lgb

from copilot import config


def rmse(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(pred) - np.asarray(true)) ** 2)))


def nasa_score(true: np.ndarray, pred: np.ndarray) -> float:
    """The C-MAPSS asymmetric score. Lower is better, and late is worse than early.

    A late prediction (pred > true) means the engine fails before you planned to
    pull it, so it is penalised on a shorter exponential constant than an early one.
    """
    d = np.asarray(pred, dtype=float) - np.asarray(true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def tier(rul: np.ndarray | pd.Series) -> np.ndarray:
    """Risk tier from predicted RUL. Thresholds live in config."""
    rul = np.asarray(rul, dtype=float)
    return np.where(rul < config.TIER_RED, "Red", np.where(rul <= config.TIER_AMBER, "Amber", "Green"))


def train_ridge(x: pd.DataFrame, y: pd.Series) -> Ridge:
    model = Ridge(alpha=1.0, random_state=config.SEED)
    model.fit(x, y)
    return model


def train_gbm(
    x: pd.DataFrame,
    y: pd.Series,
    x_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    rounds: dict[str, int] | None = None,
) -> dict[str, lgb.Booster]:
    """An L2 point model plus one model per quantile level.

    The quantile objective optimises pinball loss, which is not what RMSE rewards, so
    the headline number comes from a squared-error model and the quantile models only
    carry the interval. Keeping them separate costs one extra booster.

    Pass a validation set to pick the number of rounds by early stopping, or pass
    `rounds` to reuse a previously chosen count. Cross-validation folds use the second
    form so they do not need a validation split of their own.
    """
    if (x_val is None) == (rounds is None):
        raise ValueError("pass either a validation set or a fixed round count, not both or neither")

    objectives = {"point": {"objective": "regression"}}
    objectives.update({f"q{int(q * 100)}": {"objective": "quantile", "alpha": q} for q in config.QUANTILES})

    models: dict[str, lgb.Booster] = {}
    for name, objective in objectives.items():
        early = x_val is not None
        models[name] = lgb.train(
            {**config.GBM_PARAMS, **objective},
            lgb.Dataset(x, label=y),
            num_boost_round=config.GBM_MAX_ROUNDS if early else rounds[name],
            valid_sets=[lgb.Dataset(x_val, label=y_val)] if early else None,
            callbacks=[lgb.early_stopping(config.GBM_EARLY_STOPPING, verbose=False)] if early else [],
        )
    return models


def predict_gbm(models: dict[str, lgb.Booster], x: pd.DataFrame) -> pd.DataFrame:
    """Point prediction plus a lower and upper bound, clipped to a sane RUL range."""
    out = pd.DataFrame(index=x.index)
    for name, model in models.items():
        out[name] = np.clip(model.predict(x), 0, config.RUL_CAP)
    # Quantile models are fitted independently and can cross on individual rows.
    qcols = [c for c in out.columns if c.startswith("q")]
    out[qcols] = np.sort(out[qcols].to_numpy(), axis=1)
    return out


def conformal_width(true: np.ndarray, preds: pd.DataFrame, alpha: float = 0.2) -> float:
    """How far the raw quantile band has to be widened to actually hold (1 - alpha).

    Conformalised quantile regression: the conformity score is how far outside the
    band each held-out engine fell, and the band is widened by the (1 - alpha)
    empirical quantile of those scores. Raw LightGBM quantiles are overconfident
    because they are fitted, not calibrated; this is the correction, computed on
    validation engines the models never saw.
    """
    true = np.asarray(true, dtype=float)
    scores = np.maximum(preds["q10"].to_numpy() - true, true - preds["q90"].to_numpy())
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level, method="higher"))


def apply_conformal(preds: pd.DataFrame, width: float) -> pd.DataFrame:
    preds = preds.copy()
    preds["lower"] = np.clip(preds["q10"] - width, 0, config.RUL_CAP)
    preds["upper"] = np.clip(preds["q90"] + width, 0, config.RUL_CAP)
    return preds


def top_attributions(model: lgb.Booster, x: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """Per-prediction TreeSHAP contributions, reduced to the k strongest drivers."""
    contrib = model.predict(x, pred_contrib=True)[:, :-1]  # last column is the base value
    order = np.argsort(-np.abs(contrib), axis=1)[:, :k]
    names = np.asarray(x.columns)
    out = pd.DataFrame(index=x.index)
    for rank in range(k):
        idx = order[:, rank]
        out[f"driver{rank + 1}"] = names[idx]
        out[f"driver{rank + 1}_contrib"] = contrib[np.arange(len(idx)), idx]
    return out


# --------------------------------------------------------------------------- deep model

def make_windows(
    frame: pd.DataFrame,
    sensors: list[str],
    window: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Sequence windows per engine, front-padded so early cycles are still usable.

    Returns one window per row, shaped (rows, sensors, window), so a prediction exists
    at every cycle rather than only after the first full window.
    """
    blocks, index = [], []
    for unit, group in frame.groupby("unit", sort=True):
        values = group[sensors].to_numpy(dtype=np.float32)
        padded = np.vstack([np.repeat(values[:1], window - 1, axis=0), values])
        strided = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)
        blocks.append(strided.transpose(0, 1, 2))
        index.append(group.index.to_numpy())
    return np.concatenate(blocks), np.concatenate(index)


class CNN(torch.nn.Module):
    """Small 1D-CNN over the sensor window. Deliberately plain: this is the comparison
    model, and a bigger one would take longer to train without changing the argument."""

    def __init__(self, n_sensors: int, window: int = 30):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(n_sensors, 32, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 32, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * window, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_cnn(
    x: np.ndarray,
    y: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 30,
    batch_size: int = 256,
) -> CNN:
    """Train on CPU with a fixed seed, keeping the weights that scored best on validation."""
    torch.manual_seed(config.SEED)
    model = CNN(n_sensors=x.shape[1], window=x.shape[2])
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    tensors = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.float32)))
    loader = torch.utils.data.DataLoader(tensors, batch_size=batch_size, shuffle=True)
    val_x = torch.from_numpy(x_val)
    val_y = torch.from_numpy(y_val.astype(np.float32))

    best_loss, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimiser.zero_grad()
            loss_fn(model(batch_x), batch_y).backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y))
        if val_loss < best_loss:
            best_loss, best_state = val_loss, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def predict_cnn(model: CNN, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            out.append(model(torch.from_numpy(x[start : start + batch_size])).numpy())
    return np.clip(np.concatenate(out), 0, config.RUL_CAP)
