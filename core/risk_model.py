RISK_DIR = "data/risk_model"

import logging

import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr
from sklearn.preprocessing import LabelEncoder

from core.market_data import MarketData

logger = logging.getLogger(__name__)


def _latest_fit_map(loadings_dates):
    return (
        pd.Series(loadings_dates, pd.PeriodIndex(loadings_dates, freq="B"))
        .resample("B")
        .ffill()
    )


class RiskModel(object):
    def __init__(self):
        self._fitmap = None
        self.srisk = None
        self._loadings = None
        self.load_data()

    def load_data(self):
        srisk = read_risk_data("srisk")
        srisk.index = pd.PeriodIndex(srisk.index, freq="B", name="date")
        self.srisk = srisk

        self._loadings = lds = (
            read_risk_data("loadings")
            .set_index(["date", "ticker", "factor"])
            .squeeze()
            .to_xarray()
        )
        self._fitmap = _latest_fit_map(lds.indexes["date"])

    def get_latest_loadings(self, dt):
        return self._loadings.sel(date=self._fitmap[dt]).to_pandas().fillna(0)

    def to_risk_exposures(self, weights):
        """
        takes time-series of weights to time-series of risk exposures
        """
        reindexed = weights.reindex(self.srisk.index).fillna(0)
        fexp = {}
        for d in reindexed.index:
            lds = self.get_latest_loadings(d)
            fexp[d] = lds.T.dot(reindexed.loc[d])
        fexp = pd.DataFrame(fexp).T
        srisk = self.srisk * reindexed
        return fexp.join(srisk).fillna(0)


def init_model(n_factors, n_dates, n_tickers):
    """
    creates new network to learn latent factors and returns
    from historical returns data
    """
    date = tf.keras.Input((1,), name="date", dtype="int32")
    ticker = tf.keras.Input((1,), name="ticker", dtype="int32")

    #  learnable table of date -> factor returns
    date_embedded = tf.keras.layers.Embedding(
        n_dates, n_factors, name="date_embedding"
    )(date)

    #  learnable table of ticker -> factor loadings
    ticker_embedded = tf.keras.layers.Embedding(
        n_tickers, n_factors, name="ticker_embedding"
    )(ticker)

    pred = tf.keras.layers.Reshape((1,))(
        tf.keras.layers.Dot(axes=-1)([date_embedded, ticker_embedded])
    )

    model = tf.keras.Model(inputs=[date, ticker], outputs=pred)
    model.compile("Adagrad", "mse")
    return model


def update_model(model, data, encoders, epochs, batch_size):
    """
    updates a (potentially already trained) model for the new data
    returning rotated risk loadings
    :param encoders: a dictionary of LabelEncoders for dates and tickers
    """
    model.fit(
        dict(data.drop("total_return", axis=1)),
        data.total_return,
        epochs=epochs,
        batch_size=batch_size,
        verbose=False,
    )

    factors = pd.DataFrame(model.get_layer("date_embedding").get_weights()[0]).reindex(
        data.date.unique()
    )

    loadings = pd.DataFrame(
        model.get_layer("ticker_embedding").get_weights()[0]
    ).reindex(data.ticker.unique())

    loadings.index = encoders["ticker"].inverse_transform(loadings.index)

    #  rotating loadings so the factors are uncorrelated and unit variance
    rotated_loadings = loadings.dot(np.linalg.cholesky(factors.cov()))
    rotated_loadings.index.name = "ticker"
    rotated_loadings.columns.name = "factor"
    return rotated_loadings


def fit_model(n_factors, epochs, retrain_freq, batch_size):
    md = MarketData()
    X = md.total_return.stack().reset_index(name="total_return")

    input_cols = X.columns.difference(["total_return"])
    encoders = {c: LabelEncoder().fit(X[c]) for c in input_cols}

    for c in encoders:
        X[c] = encoders[c].transform(X[c])

    n_dates = len(encoders["date"].classes_)
    n_tickers = len(encoders["ticker"].classes_)

    model = init_model(n_factors, n_dates, n_tickers)
    loadings = {}
    t = retrain_freq

    while t < n_dates:
        if (t % retrain_freq == 0) | (t == n_dates - 1):
            print(f"fitting for date index {t}")
            data = X[(X.date >= t - retrain_freq) & (X.date < t)]
            loadings[t] = update_model(model, data, encoders, epochs, batch_size)
        t += 1

    da = xr.Dataset(loadings).to_array("date")
    da["date"] = pd.PeriodIndex(
        encoders["date"].inverse_transform(da.indexes["date"]), freq="B"
    )

    latest_fit_map = (
        pd.Series(da.indexes["date"], da.indexes["date"]).resample("B").ffill()
    )
    latest_fit_map.index.name = "date"

    #  preparing security returns for factor extraction
    security_returns = md.total_return.reindex(latest_fit_map.index)
    security_returns = security_returns[security_returns.count(axis=1) > 100]

    factors = {}
    resids = {}
    actuals = {}
    for dt in security_returns.index:
        # TODO: vectorize these calcs to speed this up
        y = security_returns.loc[dt].dropna()
        loadings = (
            da.sel(date=latest_fit_map[dt]).to_pandas().reindex(y.index).fillna(0)
        )
        factors[dt] = frets = np.linalg.pinv(loadings).dot(y)
        resids[dt] = res = y - (loadings * frets).sum(axis=1)
        actuals[dt] = y

    def to_frame(x):
        x = pd.DataFrame(x).T
        x.index = pd.PeriodIndex(x.index, freq="B", name="date")
        return x

    resids = to_frame(resids)
    actuals = to_frame(actuals)
    factors = to_frame(factors)

    srisk = resids.ewm(span=retrain_freq).std()

    #  loadings back to tidy dataframe format
    loadings = (
        da.stack(z=["date", "ticker", "factor"])
        .to_pandas()
        .dropna()
        .reset_index(name="loading")
    )
    return dict(
        loadings=loadings,
        actuals=actuals,
        residuals=resids,
        factors=factors,
        srisk=srisk,
    )


def write_fit_results(results):
    for x in results:
        fname = f"{RISK_DIR}/{x}"
        results[x].to_csv(fname)


def read_risk_data(name):
    if name == "loadings":
        return pd.read_csv(
            f"{RISK_DIR}/{name}", usecols=["date", "ticker", "factor", "loading"]
        )

    df = pd.read_csv(f"{RISK_DIR}/{name}").set_index("date").squeeze()
    if name != "last_fit_map":
        df.columns.name = "ticker"

    return df
