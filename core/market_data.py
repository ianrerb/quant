import pandas as pd

MARKET_DATA_FILE = "quant_case_data/market_data.csv"
MIN_STOCKS = 100

COL_RENAME = {
    "Date": "date",
    "Ticker": "ticker",
    "Total Return": "total_return",
    "Market Cap": "market_cap",
    "Daily Volume": "daily_volume",
}


class MarketData:
    def __init__(self):
        self._dataset = None

    def load_data(self, reload=False):
        if (self._dataset is not None) & (~reload):
            return

        data = (
            pd.read_csv(MARKET_DATA_FILE)
            .rename(columns=COL_RENAME)
            .set_index(["date", "ticker"])
            .to_xarray()
        )

        data["date"] = pd.PeriodIndex(data["date"].values, freq="B")

        valid_dates = data.market_cap.count("ticker") > 100

        self._dataset = data.where(valid_dates).dropna("date", how="all")

    @property
    def total_return(self):
        self.load_data()
        return self._dataset.total_return.to_pandas()

    @property
    def market_cap(self):
        self.load_data()
        return self._dataset.market_cap.to_pandas()

    @property
    def daily_volume(self):
        self.load_data()
        return self._dataset.daily_volume.to_pandas()
