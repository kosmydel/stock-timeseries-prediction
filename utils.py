from darts.metrics import mape, mse, rmse, mae
import json
from tqdm.notebook import tqdm
from typing import Callable, List
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts import TimeSeries
import matplotlib.pyplot as plt
import time
import pickle

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


class Dataset:
    def __init__(self, series: TimeSeries, name: str):
        self.series = series
        self.name = name

        self.preprocess()

    def preprocess(self):
        pass


class TimeseriesExperiment:
    # TODO: Add metric support
    def __init__(self, model: ForecastingModel, dataset: Dataset, parameters: dict, forecast_horizon: int = 3):
        self.model = model
        self.dataset = dataset
        self.parameters = parameters
        self.forecast_horizon = forecast_horizon

    def load_or_find_parameters(self):
        model_name = f'{self.dataset.name}_{self.model.__class__.__name__}_{self.forecast_horizon}.json'
        model = load_model(model_name)
        if model is None:
            self.find_parameters()
            save_model(model_name, self.model)
        else:
            self.model = model

    def find_parameters(self):
        model, parameters, metric = self.model.gridsearch(self.parameters, self.dataset.series, verbose=True)

        self.model = model
        print('Best parameters:', parameters, 'Metric:', metric)

    def run(self):
        self.load_or_find_parameters()

        result = self.model.historical_forecasts(self.dataset, forecast_horizon=self.forecast_horizon)

        metrics = calculate_metrics(self.dataset, result)
        metrics['model'] = model.__class__.__name__
        metrics['forecast_horizon'] = self.forecast_horizon
        metrics['dataset'] = self.dataset
        metrics['experiment_time'] = time.time()

        file_name = f'{RESULTS_PATH}{self.dataset}_{model.__class__.__name__}_{self.forecast_horizon}.json'

        with open(file_name, 'w') as f:
            json.dump(metrics, f)



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


def save_model(file_name: str, model) -> None:
    with open(file_name, 'wb') as f:
        pickle.dump(model, f)


def load_model(file_name) -> ForecastingModel | None:
    try:
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

if __name__ == '__main__':
    model = load_model('model.pkl')