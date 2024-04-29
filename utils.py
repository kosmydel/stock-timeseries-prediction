from darts.metrics import mape, mse, rmse, mae
import json
from tqdm.notebook import tqdm
from typing import List
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts import TimeSeries
import matplotlib.pyplot as plt
import time

RESULTS_PATH = 'results/'

def calculate_metrics(series, forecast):
    metrics = {}
    metrics['mape'] = mape(series, forecast)
    metrics['mse'] = mse(series, forecast)
    metrics['rmse'] = rmse(series, forecast)
    metrics['mae'] = mae(series, forecast)
    return metrics

def print_metrics(name, series, forecast):
    metrics = calculate_metrics(series, forecast)
    print(name + ' metrics:')
    for m, v in metrics.items():
        print(f'{m}: {v:.2f}')
    print('')

def plot_forecast(series, forecast, title):
    plt.figure(figsize=(25,5))
    series.plot(label="actual")
    forecast.plot(label="forecast")
    plt.title(title)
    plt.legend()
    plt.show()


def run_experiment():
    print('Running experiment')


def backtest(models: List[ForecastingModel], series: TimeSeries, dataset: str, forecast_horizon: int = 3):
    """
    Run backtest for a list of models
    :param models: list of trained models
    :param series: TimeSeries
    """

    for model in models:
        result = model.historical_forecasts(series, forecast_horizon=forecast_horizon)

        metrics = calculate_metrics(series, result)
        metrics['model'] = model.__class__.__name__
        metrics['forecast_horizon'] = forecast_horizon
        metrics['dataset'] = dataset
        metrics['experiment_time'] = time.time()

        file_name = f'{RESULTS_PATH}{dataset}_{model.__class__.__name__}_{forecast_horizon}.json'

        with open(file_name, 'w') as f:
            json.dump(metrics, f)



if __name__ == '__main__':
    print('This is the main program')
