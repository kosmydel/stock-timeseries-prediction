from darts.metrics import mape, mse, rmse, mae
import json
from typing import List, Sequence
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts import TimeSeries
import matplotlib.pyplot as plt
import time
import pickle
from sklearn.preprocessing import StandardScaler
from darts.dataprocessing.transformers.scaler import Scaler
import os
import glob
import pandas as pd
from darts.utils.model_selection import train_test_split

RESULTS_PATH = "results/"

HORIZONS = [1, 2, 3, 5, 7, 9, 10]


def calculate_metrics(series: TimeSeries, forecast: TimeSeries):
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
    def __init__(
        self,
        series: TimeSeries | List[TimeSeries],
        name: str,
        past_covariates: TimeSeries | List[TimeSeries] | None = None,
        future_covariates: TimeSeries | List[TimeSeries] | None = None,
        preprocess=True,
        test_size=0.2,
    ):
        self.series_unscaled = series
        self.name = name
        self.past_covariates_unscaled = past_covariates
        self.future_covariates_unscaled = future_covariates

        # split series into train and test sets
        self.train_unscaled, self.test_unscaled = train_test_split(series, test_size, axis=1)

        print(f"Train size: {len(self.train_unscaled)} Test size: {len(self.test_unscaled)}")

        if past_covariates is not None:
            self.past_covariates_train_unscaled, self.past_covariates_test_unscaled = (
                train_test_split(past_covariates, test_size)
            )

        if preprocess:
            self.preprocess()

    def preprocess(self):
        self.scaler = StandardScaler()
        self.transformer = Scaler(self.scaler, global_fit=True)

        print(self.series_unscaled[0])
        print(self.train_unscaled[0])
        print(self.test_unscaled[0])

        self.series = self.transformer.fit_transform(self.series_unscaled)
        self.train = self.transformer.transform(self.train_unscaled)
        self.test = self.transformer.transform(self.test_unscaled)

        # Past
        self.scaler_covariates = StandardScaler()
        self.transformer_covariates = Scaler(self.scaler_covariates, global_fit=True)

        if self.past_covariates_unscaled is not None:
            self.past_covariates = self.transformer_covariates.fit_transform(
                self.past_covariates_unscaled
            )

            self.past_covariates_train = self.transformer_covariates.transform(
                self.past_covariates_train_unscaled
            )
            self.past_covariates_test = self.transformer_covariates.transform(
                self.past_covariates_test_unscaled
            )
        else:
            self.past_covariates = None
            self.past_covariates_train = None
            self.past_covariates_test = None

        # Future
        self.scaler_future_covariates = StandardScaler()
        self.transformer_future_covariates = Scaler(self.scaler_future_covariates, global_fit=True)

        if self.future_covariates_unscaled is not None:
            self.future_covariates = self.transformer_future_covariates.fit_transform(
                self.future_covariates_unscaled
            )
        else:
            self.future_covariates = None

    def postprocess(self, series):
        return self.transformer.inverse_transform(series)

    def plot_train_test(self):
        plt.figure(figsize=(25, 5))
        if isinstance(self.train, TimeSeries):
            self.train.plot(label="train")
            self.test.plot(label="test")
        else:
            # TODO: plot multiple series
            pass
        plt.title(self.name)
        plt.legend()
        plt.show()


class TimeseriesExperiment:
    def __init__(
        self,
        model: ForecastingModel,
        dataset: Dataset,
        parameters: dict = {},
        forecast_horizon: int = 1,
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
        params = {}
        if self.model.supports_past_covariates:
            params["past_covariates"] = self.dataset.past_covariates_train
        if self.model.supports_future_covariates:
            params["future_covariates"] = self.dataset.future_covariates

        # assert type(self.dataset.train) in [TimeSeries, List[TimeSeries]]

        if len(self.parameters) == 0:
            self.trained_model = self.model.fit(self.dataset.train, **params)
            print("No parameters to search")
            return None
        else:
            print("Searching for best parameters", self.parameters)
            model, parameters, metric = self.model.gridsearch(
                self.parameters,
                self.dataset.train,
                forecast_horizon=self.forecast_horizon,
                **params,
            )
            self.trained_model = model
            self.trained_model.fit(self.dataset.train, **params)
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
        # Load or train model
        self.load_or_train()

        params = {}
        assert self.trained_model is not None

        if self.trained_model.supports_past_covariates:
            params["past_covariates"] = self.dataset.past_covariates
        if self.trained_model.supports_future_covariates:
            params["future_covariates"] = self.dataset.future_covariates

        start_time = self.dataset.test.start_time() if isinstance(self.dataset.test, TimeSeries) else self.dataset.test[0].start_time()

        print(start_time)

        # Measure model performance using historical forecasts
        result = self.trained_model.historical_forecasts(
            self.dataset.series,
            start=start_time,
            forecast_horizon=self.forecast_horizon,
            retrain=self.retrain,
            **params,
        )

        result_unscaled = self.dataset.postprocess(result)

        # Plot forecast
        plot_forecast(
            self.dataset.test_unscaled, result_unscaled, f"{self.model.__class__.__name__}"
        )

        # Calculate metrics
        metrics = calculate_metrics(self.dataset.test_unscaled, result_unscaled)
        metrics["model"] = self.model.__class__.__name__
        metrics["forecast_horizon"] = self.forecast_horizon
        metrics["dataset"] = self.dataset.name
        metrics["experiment_time"] = time.time()
        metrics["parameters"] = self.trained_model._model_params

        # Save results
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
            print(f"Model: {model.__class__.__name__} Metrics: {metrics}")
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


def read_results(directory="results") -> pd.DataFrame:
    frames = []
    for file in glob.glob(f"{directory}/*.json"):
        with open(file) as f:
            data = pd.read_json(f, typ="series", orient="index")
            frames.append(data)
    res = pd.concat(frames, axis=1).T
    return res


if __name__ == "__main__":
    model = load_model("model.pkl")
