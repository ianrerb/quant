import numpy as np
import pandas as pd
import xarray as xr

from core.market_data import MarketData
from core.risk_model import RiskModel, read_risk_data

ALPHA_DIR = "data/alpha"


def rank_signal(raw, shift=1):
    sig = raw.rank(axis=1)
    sig = sig.sub(sig.mean(axis=1), axis=0)
    sig = sig.div(sig.abs().sum(axis=1), axis=0)
    return sig.shift(shift)


def compute_short_term_reversal():
    resids = read_risk_data("residuals")
    srisk = read_risk_data("srisk")
    sig = (-resids / srisk).ewm(span=5, min_periods=2).mean()
    sig.index = pd.PeriodIndex(sig.index, freq="B")
    return rank_signal(sig)


def compute_adv_signal():
    data = MarketData()
    return rank_signal(-data.daily_volume.rolling(30, 10).median())


configs = {
    "reversal": (compute_short_term_reversal, 0.7),
    "adv": (compute_adv_signal, 0.99),
}


def run_strat(name):
    conf = configs[name]
    expos = RiskModel().to_risk_exposures(conf[0]())
    expos = expos.div((expos ** 2).sum(axis=1) ** 0.5, axis=0)
    expos.index.name = "date"
    expos.to_csv(f"{ALPHA_DIR}/{name}_normed_exposures")

    returns = (conf[0]() * MarketData().total_return).sum(axis=1)
    decay = conf[1]

    w = returns.ewm(span=252, min_periods=252)
    sharpe = (w.mean() / w.std()).shift(2).clip_lower(0)
    horizon_alpha = sharpe * 1.0 / (1.0 - decay) / (1.0 - sharpe * decay)

    perf = pd.DataFrame(
        {"sharpe": sharpe, "returns": returns, "horizon_alpha": horizon_alpha}
    )
    perf.index.name = "date"
    perf.to_csv(f"{ALPHA_DIR}/{name}_perf")


def run_risk_alphas(strat_list):
    horizon_alphas = {}
    exposures = {}

    for strat in strat_list:
        perf = pd.read_csv(f"data/alpha/{strat}_perf").set_index("date")
        horizon_alphas[strat] = perf.horizon_alpha
        expos = pd.read_csv(f"data/alpha/{strat}_normed_exposures").set_index("date")
        expos.columns.name = "factor"
        exposures[strat] = expos

    exposures = (
        xr.Dataset(exposures)
        .to_array("signal")
        .transpose("date", "signal", "factor")
        .fillna(0)
    )
    horizon_alphas = (
        xr.Dataset(horizon_alphas)
        .to_array("signal")
        .transpose("date", "signal")
        .fillna(0)
    )

    result = pd.DataFrame(
        np.einsum(
            "abc,ac->ab",
            np.linalg.pinv(exposures),
            horizon_alphas.reindex(date=exposures.date),
        ),
        exposures.date.to_index(),
        exposures.factor.to_index(),
    )

    result.to_csv(f"{ALPHA_DIR}/risk_alphas")


def read_risk_alphas():
    alphas = pd.read_csv(f"{ALPHA_DIR}/risk_alphas")
    alphas = alphas.set_index("date")
    alphas.index = pd.PeriodIndex(alphas.index, freq="B", name="date")
    return alphas[alphas.abs().max(axis=1) > 0]


def run():
    strats = configs.keys()
    for strat in strats:
        run_strat(strat)

    run_risk_alphas(strats)
