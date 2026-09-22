"""C-MAPSS download, parsing, RUL labelling and feature pipeline.

The dataset is simulated turbofan degradation from the NASA Prognostics Center of
Excellence. Training trajectories run to failure; test trajectories are truncated
some unknown number of cycles before failure, and the true remaining life is given
in a separate file.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd

from copilot import config

SENSORS = [f"s{i}" for i in range(1, 22)]
OP_SETTINGS = ["op1", "op2", "op3"]
COLUMNS = ["unit", "cycle", *OP_SETTINGS, *SENSORS]


def download() -> None:
    """Fetch and extract the C-MAPSS text files. No-op if they are already there."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    expected = config.DATA_DIR / f"train_{config.DATASET}.txt"
    if expected.exists():
        return

    print(f"downloading {config.DATA_URL}")
    with urllib.request.urlopen(config.DATA_URL, timeout=300) as response:
        outer_bytes = response.read()

    # The NASA archive wraps a second zip that holds the actual text files.
    with zipfile.ZipFile(io.BytesIO(outer_bytes)) as outer:
        inner_name = next(n for n in outer.namelist() if n.endswith("CMAPSSData.zip"))
        with zipfile.ZipFile(io.BytesIO(outer.read(inner_name))) as inner:
            for name in inner.namelist():
                if name.endswith(".txt"):
                    (config.DATA_DIR / name).write_bytes(inner.read(name))

    if not expected.exists():
        raise RuntimeError(f"download finished but {expected} is missing")


def load(split: str) -> pd.DataFrame:
    """Read train_FD001.txt or test_FD001.txt into a typed frame."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    path = config.DATA_DIR / f"{split}_{config.DATASET}.txt"
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)
    return df


def load_true_rul() -> pd.Series:
    """True remaining cycles for each test engine at its last observed cycle."""
    path = config.DATA_DIR / f"RUL_{config.DATASET}.txt"
    values = pd.read_csv(path, header=None).iloc[:, 0].astype(int)
    values.index = np.arange(1, len(values) + 1)  # unit ids are 1-based and in order
    values.index.name = "unit"
    return values.rename("true_rul")


def add_rul(df: pd.DataFrame, true_rul: pd.Series | None = None) -> pd.DataFrame:
    """Attach the piecewise-linear RUL label, capped at config.RUL_CAP.

    Training engines run to failure, so RUL is cycles until the last row. Test engines
    stop early, so their remaining life at any cycle is offset by the true RUL given
    for the final observed cycle. The cap reflects that degradation is not observable
    while an engine is healthy: predicting 'more than 125 cycles left' is the honest
    answer, and a linear label there would only teach the model to fit noise.
    """
    df = df.copy()
    last_cycle = df.groupby("unit")["cycle"].transform("max")
    offset = 0 if true_rul is None else df["unit"].map(true_rul)
    df["rul"] = last_cycle - df["cycle"] + offset
    df["rul_capped"] = df["rul"].clip(upper=config.RUL_CAP)
    return df


def useful_sensors(train: pd.DataFrame) -> list[str]:
    """Sensors that actually vary. In FD001 seven of the 21 are flat and carry nothing."""
    keep = [s for s in SENSORS if train[s].std() > 1e-6 and train[s].nunique() > 2]
    if len(keep) < 10:
        raise RuntimeError(f"only {len(keep)} usable sensors found, pipeline is wrong")
    return keep


def fit_stats(train: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    """Normalization statistics, computed on training engines only."""
    return pd.DataFrame({"mean": train[sensors].mean(), "std": train[sensors].std()})


def featurize(df: pd.DataFrame, sensors: list[str], stats: pd.DataFrame) -> pd.DataFrame:
    """Per-row features: normalized value, rolling mean and spread, and a trend term.

    Everything is computed within an engine, so no information crosses unit boundaries.
    """
    out = pd.DataFrame({"unit": df["unit"].values, "cycle": df["cycle"].values}, index=df.index)
    z = (df[sensors] - stats["mean"]) / stats["std"]
    z.insert(0, "unit", df["unit"].values)
    grouped = z.groupby("unit")[sensors]

    short, long = config.ROLLING_WINDOWS
    means = {w: grouped.rolling(w, min_periods=1).mean().reset_index(level=0, drop=True) for w in (short, long)}
    sd_long = grouped.rolling(long, min_periods=2).std().reset_index(level=0, drop=True)

    for s in sensors:
        out[s] = z[s]
        out[f"{s}_m{short}"] = means[short][s]
        out[f"{s}_m{long}"] = means[long][s]
        out[f"{s}_sd{long}"] = sd_long[s]
        # Two-point slope on the smoothed signal, not least squares. The rolling
        # means already denoise it. Swap in a regression slope if the trend
        # features turn out to be weak in the attribution charts.
        smoothed = means[short][s]
        out[f"{s}_d{config.SLOPE_WINDOW}"] = smoothed - smoothed.groupby(df["unit"]).shift(config.SLOPE_WINDOW)

    return out.fillna(0.0)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in ("unit", "cycle")]


def sensor_of(feature: str) -> str:
    """Map a feature name such as 's11_m20' back to its sensor, 's11'.

    Lives here rather than in models so the app can decode a feature name without
    importing the training stack. Loading torch, lightgbm and scikit-learn into the
    Streamlit process to split a string is two seconds nobody needs to wait for.
    """
    return feature.split("_")[0]


def split_engines(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Hold out whole engines for validation. Splitting by row would leak."""
    units = np.sort(train["unit"].unique())
    rng = np.random.default_rng(config.SEED)
    val = rng.choice(units, size=config.VAL_ENGINES, replace=False)
    fit = np.setdiff1d(units, val)
    if set(fit) & set(val):
        raise RuntimeError("train and validation engines overlap")
    return fit, np.sort(val)


# C-MAPSS sensor descriptions, from the dataset readme. Used to give the work order
# generator something more meaningful to cite than "s11".
SENSOR_DESCRIPTIONS = {
    "s1": "T2, total temperature at fan inlet",
    "s2": "T24, total temperature at LPC outlet",
    "s3": "T30, total temperature at HPC outlet",
    "s4": "T50, total temperature at LPT outlet",
    "s5": "P2, pressure at fan inlet",
    "s6": "P15, total pressure in bypass duct",
    "s7": "P30, total pressure at HPC outlet",
    "s8": "Nf, physical fan speed",
    "s9": "Nc, physical core speed",
    "s10": "epr, engine pressure ratio",
    "s11": "Ps30, static pressure at HPC outlet",
    "s12": "phi, ratio of fuel flow to Ps30",
    "s13": "NRf, corrected fan speed",
    "s14": "NRc, corrected core speed",
    "s15": "BPR, bypass ratio",
    "s16": "farB, burner fuel-air ratio",
    "s17": "htBleed, bleed enthalpy",
    "s18": "Nf_dmd, demanded fan speed",
    "s19": "PCNfR_dmd, demanded corrected fan speed",
    "s20": "W31, HPT coolant bleed flow",
    "s21": "W32, LPT coolant bleed flow",
}
