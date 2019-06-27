import logging

from core import risk_model, signals, optimizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def fit_risk_model():
    """
    rebuild risk data in data/risk_model/
    """
    results = risk_model.fit_model(
        n_factors=10,
        epochs=50,
        retrain_freq=60,
        batch_size=10000)
    risk_model.write_fit_results(results)

def run(refit_risk_model=True):
    if refit_risk_model:
        logger.info('fitting risk model...')
        fit_risk_model()

    logger.info('running signal generation...')
    signals.run()

    logger.info('running backtest...')
    optimizer.run_backtest()

    logger.info('complete')
