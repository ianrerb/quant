import numpy as np
import pandas as pd
from mosek.fusion import Domain, Expr, Model, ObjectiveSense

from core.risk_model import RiskModel
from core.signals import read_risk_alphas

DEFAULT_COST = 5e-4


def run_backtest(sample=None):
    risk_alphas = read_risk_alphas().dropna(how="all").fillna(0)
    rmodel = RiskModel()
    init = np.zeros(len(rmodel.srisk.columns))

    valid_dates = risk_alphas.index.intersection(rmodel.srisk.index)
    if sample:
        valid_dates = valid_dates[-sample:]

    weights = {}
    for dt in valid_dates:
        print(dt)
        try:
            w = solve(
                init,
                risk_alphas.loc[dt],
                rmodel.get_latest_loadings(dt),
                rmodel.srisk.loc[dt],
            )
            weights[dt] = w
            init = w
        except Exception as e:
            print(str(e))
            weights[dt] = init

    weights = pd.DataFrame(weights).T
    weights.index.name = 'date'
    weights.to_csv("data/portfolio/weights.csv")

    formatted = (
        weights.stack()
        .reset_index(name="Weight")
        .rename(columns={"date": "Date", "ticker": "Ticker"})
        .dropna()
    )

    formatted.to_csv("final_portfolio.csv")


def solve(x0, risk_alphas, loadings, srisk, cost_per_trade=DEFAULT_COST, max_risk=0.01):
    N = len(x0)
    #  don't hold no risk data (likely dead)
    lim = np.where(srisk.isnull(), 0.0, 1.0)
    loadings = loadings.fillna(0)
    srisk = srisk.fillna(0)
    risk_alphas = risk_alphas.fillna(0)

    with Model() as m:
        w = m.variable(N, Domain.inRange(-lim, lim))
        longs = m.variable(N, Domain.greaterThan(0))
        shorts = m.variable(N, Domain.greaterThan(0))
        gross = m.variable(N, Domain.greaterThan(0))

        m.constraint(
            "leverage_consistent",
            Expr.sub(gross, Expr.add(longs, shorts)),
            Domain.equalsTo(0),
        )

        m.constraint(
            "net_consistent", Expr.sub(w, Expr.sub(longs, shorts)), Domain.equalsTo(0.0)
        )

        m.constraint("leverage_long", Expr.sum(longs), Domain.lessThan(1.0))

        m.constraint("leverage_short", Expr.sum(shorts), Domain.lessThan(1.0))

        buys = m.variable(N, Domain.greaterThan(0))
        sells = m.variable(N, Domain.greaterThan(0))

        gross_trade = Expr.add(buys, sells)
        net_trade = Expr.sub(buys, sells)
        total_gross_trade = Expr.sum(gross_trade)

        m.constraint(
            "net_trade",
            Expr.sub(w, net_trade),
            Domain.equalsTo(np.asarray(x0)),  #  cannot handle series
        )

        #  add risk constraint
        vol = m.variable(1, Domain.lessThan(max_risk))
        stacked = Expr.vstack(vol.asExpr(), Expr.mulElm(w, srisk.values))
        stacked = Expr.vstack(stacked, Expr.mul(loadings.values.T, w))
        m.constraint("vol-cons", stacked, Domain.inQCone())

        alphas = risk_alphas.dot(np.vstack([loadings.T, np.diag(srisk)]))

        gain = Expr.dot(alphas, net_trade)
        loss = Expr.mul(cost_per_trade, total_gross_trade)
        m.objective(ObjectiveSense.Maximize, Expr.sub(gain, loss))

        m.solve()
        result = pd.Series(w.level(), srisk.index)
        return result
