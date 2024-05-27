# Predicting the Future: A Comparison of Modern TimeSeries Models

## Introduction

Navigating the world of TimeSeries prediction models can be a maze. With numerous leading-edge models at play, knowing which offer the best predictive performance, require the least parameter tuning, and thrive under certain conditions, is invaluable. This post demystifies that maze. We'll put several state-of-the-art models head-to-head, testing them against three disparate datasets, to provide clear, practical insights that guide in selecting the right tool for your TimeSeries prediction needs. Strap in and join us on this data-driven journey!

## Datasets

### Walmart Sales Forecasting

The dataset employed for this study was sourced from the [Kaggle competition - Walmart Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting), with our focus centered on predicting sales for one select store and one particular department. The dataset encapsulates weekly sales data for each store complimented with several prominent features, among which, we have strategically chosen to engage with the ones listed below:

Past Covariates:

- **Temperature** - Represents the average regional temperature
- **Fuel_Price** - Indicates the regional cost of fuel
- **CPI** - Outline of the Consumer Price Index
- **Unemployment** - Denotes the regional unemployment rate

Future Covariate:

- **IsHoliday** - Specifies if a given week is a special holiday week (This feature is known in the future)

We subsequently partitioned this dataset, creating distinct training and test sets for our analysis.

### Eletricity Consumption Forecasting

### Bitcoin Price Forecasting

// TODO: List the datasets, explain why each dataset is different

## Models

// TODO: List models, table the results

### TimeGPT

We decided to put the cutting-edge TimeGPT model to the test. A transformer-based model tailored for TimeSeries forecasting, TimeGPT is a variant of the widely appreciated GPT model, known for its efficacy in the Natural Language Processing (NLP) domain. We ran a series of tests using the Electricity Consumption Forecasting dataset as our playground.

However, the outcome was less than impressive. Not only did the performance fail to meet our expectations, but the cost linked to it was also quite high. The free trial, which amounts to a hefty sum of $1000, was almost entirely consumed during these tests. Consequently, due to its prohibitive price and underwhelming performance, the TimeGPT model was eliminated from our final comparison.

## Comparison

We have presented a table below that encapsulates the outcomes of our data analysis. This presentation aims to offer a lucid depiction of each model's performance when applied to three diverse datasets.

### Mean Squared Error (MSE) Scores of one step ahead predictions

|          | Electricity | Walmart Sales | Bitcoin |
| -------- | ----------- | ------------- | ------- |
| Baseline | 0.0075      |               |         |
| XGBoost  | 0.0086      |               |         |
| LightGBM |             |               |         |
| Prophet  | 0.0800      |               |         |
| Arima    | 0.0067      |               |         |
| TFT      |             |               |         |

## Conclusion

// TODO: Summarize the results, provide recommendations
