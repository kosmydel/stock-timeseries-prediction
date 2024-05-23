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

## Conclusion

// TODO: Summarize the results, provide recommendations
