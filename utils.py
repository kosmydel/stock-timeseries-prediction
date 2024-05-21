from darts.metrics import mape, mse, rmse, mae
import json
from tqdm.notebook import tqdm
from typing import Callable, List
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts import TimeSeries
import matplotlib.pyplot as plt
import time
import pickle
from sklearn.preprocessing import StandardScaler
from darts.dataprocessing.transformers.scaler import Scaler
import os

RESULTS_PATH = "results/"

HORIZONS = [1, 2, 3, 5, 7, 9, 10]

def calculate_metrics(series, forecast):
    metrics = {}
    metrics["mape"] = mape(series, forecast)
    metrics["mse"] = mse(series, forecast)
    metrics["rmse"] = rmse(series, forecast)
    metrics["mae"] = mae(series, forecast)
    return metrics


def print_metrics(name, series, forecast):
    metrics = calculate_metrics(series, forecast)
    print(name + " metrics:")
    for m, v in metrics.items():
        print(f"{m}: {v:.2f}")
    print("")


def plot_forecast(series, forecast, title):
    plt.figure(figsize=(25, 5))
    series.plot(label="actual")
    forecast.plot(label="forecast")
    plt.title(title)
    plt.legend()
    plt.show()


class Dataset:
    def __init__(self, series: TimeSeries, name: str, past_covariates: TimeSeries | None = None, preprocess=True):
        self.series = series
        self.name = name
        self.past_covariates = past_covariates

        # split series into train and test sets
        # this is necessaru, because split_after(0.8) removes the last point of the training set
        diff =  series[-1].time_index[0] - series[0].time_index[0]
        interval_0_8 = diff * 0.8
        split_point = series[0].time_index[0] + interval_0_8
        self.train, self.test = series.split_after(split_point)

        if past_covariates is not None:
            self.past_covariates_train, self.past_covariates_test = past_covariates.split_after(split_point)

        if preprocess:
            self.preprocess()

    def preprocess(self):
        self.scaler = StandardScaler()
        self.transformer = Scaler(self.scaler)

        self.train = self.transformer.fit_transform(self.train)
        self.test = self.transformer.transform(self.test)

        self.scaler_covariates = StandardScaler()
        self.transformer_covariates = Scaler(self.scaler_covariates)

        if self.past_covariates is not None:
            self.past_covariates_train = self.transformer_covariates.fit_transform(self.past_covariates_train)
            self.past_covariates_test = self.transformer_covariates.transform(self.past_covariates_test)

    def postprocess(self, series):
        return self.transformer.inverse_transform(series)

    def plot_train_test(self):
        plt.figure(figsize=(25, 5))
        self.train.plot(label="train")
        self.test.plot(label="test")
        plt.title(self.name)
        plt.legend()
        plt.show()


class TimeseriesExperiment:
    # TODO: Add metric support
    def __init__(
        self,
        model: ForecastingModel,
        dataset: Dataset,
        parameters: dict = {},
        forecast_horizon: int = 3,
        use_pretrained_model: bool = False,
        retrain: bool = False,
        n_last_series_from_train_in_test: int = 0,
    ):
        self.model = model
        self.dataset = dataset
        self.parameters = parameters
        self.forecast_horizon = forecast_horizon
        self.trained_model = None
        self.use_pretrained_model = use_pretrained_model
        self.retrain = retrain
        self.n_last_series_from_train_in_test = n_last_series_from_train_in_test

    def find_parameters(self):
        if len(self.parameters) == 0:
            self.trained_model = self.model.fit(self.dataset.train, past_covariates=self.dataset.past_covariates_train)
            print("No parameters to search")
            return None
        else:
            print("Searching for best parameters", self.parameters)
            model, parameters, metric = self.model.gridsearch(
                self.parameters,
                self.dataset.train,
                verbose=True,
                forecast_horizon=self.forecast_horizon,
                past_covariates=self.dataset.past_covariates_train,
            )
            self.trained_model = model
            self.trained_model.fit(self.dataset.train, past_covariates=self.dataset.past_covariates_train)
            print("Best parameters:", parameters, "Metric:", metric)
            return parameters

    def load_or_train(self):
        model_name = f"{self.dataset.name}_{self.model.__class__.__name__}.pkl"
        model_location = f"models/{model_name}"

        if os.path.exists(model_location) and self.use_pretrained_model:
            self.trained_model = load_model(model_location)
        else:
            parameters = self.find_parameters()
            save_model(model_location, self.trained_model, parameters=parameters)

    def run(self):
        self.load_or_train()

        test_set = self.dataset.test
        if self.n_last_series_from_train_in_test > 0:
            # push last n series from train to test
            test_set = self.dataset.train[-self.n_last_series_from_train_in_test:].append(self.dataset.test)

        test_covariate = self.dataset.past_covariates_test
        if self.n_last_series_from_train_in_test > 0:
            test_covariate = self.dataset.past_covariates_train[-self.n_last_series_from_train_in_test:].append(
                self.dataset.past_covariates_test
            )
        result = self.trained_model.historical_forecasts(
            test_set, forecast_horizon=self.forecast_horizon, retrain=self.retrain,
            past_covariates=test_covariate
        )

        # plot forecast
        plot_forecast(self.dataset.test, result, f"{self.model.__class__.__name__}")

        metrics = calculate_metrics(self.dataset.series, result)
        metrics["model"] = self.model.__class__.__name__
        metrics["forecast_horizon"] = self.forecast_horizon
        metrics["dataset"] = self.dataset.name
        metrics["experiment_time"] = time.time()
        metrics["parameters"] = self.trained_model._model_params

        file_name = f"{RESULTS_PATH}{self.dataset.name}_{self.model.__class__.__name__}_{self.forecast_horizon}.json"

        os.makedirs(RESULTS_PATH, exist_ok=True)

        with open(file_name, "w") as f:
            json.dump(metrics, f)

        return metrics


def backtest(
    models: List[ForecastingModel],
    series: TimeSeries,
    dataset: str,
    forecast_horizon: int = 3,
    verbose=True,
):
    """
    Run backtest for a list of models
    :param models: list of trained models
    :param series: TimeSeries
    """

    for model in models:
        result = model.historical_forecasts(series, forecast_horizon=forecast_horizon)

        metrics = calculate_metrics(series, result)
        if verbose:
            print(f'Model: {model.__class__.__name__} Metrics: {metrics}')
        metrics["model"] = model.__class__.__name__
        metrics["forecast_horizon"] = forecast_horizon
        metrics["dataset"] = dataset
        metrics["experiment_time"] = time.time()

        file_name = f"{RESULTS_PATH}{dataset}_{model.__class__.__name__}_{forecast_horizon}.json"

        with open(file_name, "w") as f:
            json.dump(metrics, f)


def save_model(file_name: str, model, parameters) -> None:
    with open(file_name, "wb") as f:
        pickle.dump(model, f)
    with open(file_name + ".params", "w") as f:
        json.dump(parameters, f)


def load_model(file_name) -> ForecastingModel | None:
    try:
        with open(file_name, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


if __name__ == "__main__":
    model = load_model("model.pkl")
