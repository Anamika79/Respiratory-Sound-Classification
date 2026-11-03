# Food Delivery Time Prediction & Late-Delivery Classification

Predicting how long an Indian food delivery order will take (regression) and flagging orders at risk of delay (classification), using **Linear Regression** and a tuned **Decision Tree**.

## Overview

This project works on a single dataset of Indian food delivery orders and solves two related tasks:

| Task | Model | Target |
|---|---|---|
| Regression | Linear Regression (OLS) | `Time_taken(min)` – delivery time in minutes |
| Classification | Decision Tree (tuned with GridSearchCV) | `Delivery_Status` – On-time / Delayed |

An order is labelled **Delayed** if its delivery time is above the median delivery time of the **training set**, and **On-time** otherwise. This gives two roughly balanced classes. Note that "Delayed" here means *slower than typical*, not late against a promised delivery time.

Both models use the same train/test split so they can be compared fairly.

## Dataset

- **Source:** [Food Delivery Dataset by Gaurav Malik (Kaggle)](https://www.kaggle.com/datasets/gauravmalik26/food-delivery-dataset)
- **Size:** about 45,000 orders from Indian cities
- **File used:** `train.csv` only (`test.csv` has no delivery times)
- **Columns:** delivery partner age and ratings, restaurant and delivery coordinates, order date and times, weather, road traffic density, vehicle condition, order type, vehicle type, multiple deliveries, festival, city type, and `Time_taken(min)`

The dataset is not included in this repository. Download `train.csv` from Kaggle and place it in the project folder (or upload it in Colab when prompted).

## Methodology

1. **Data cleaning**
   - Stripped trailing whitespace and converted the text `"NaN "` into real null values
   - Extracted the number from `Time_taken(min)` values such as `"(min) 24"`
   - Removed the `"conditions "` prefix from `Weatherconditions`
   - Converted age, ratings and multiple deliveries to numeric types
   - Removed duplicate orders
2. **Invalid records:** removed rows with zero or negative coordinates and partner ratings above 5; checked extreme values with the IQR method
3. **Feature engineering**
   - Delivery distance in km from restaurant and delivery coordinates (Haversine formula)
   - Order preparation time (pickup time minus order time, handling midnight crossover)
   - Order hour, day of week and weekend flag
4. **Train/test split:** 80/20, fixed random seed, stratified so both classes keep the same proportion
5. **Preprocessing (fitted on the training set only, to avoid data leakage)**
   - Median imputation for numeric columns, mode imputation for categorical columns
   - Ordinal encoding for `Road_traffic_density` (Low < Medium < High < Jam)
   - One-hot encoding for weather, vehicle type, order type, city and festival
   - `StandardScaler` for Linear Regression only (trees do not need scaling)
6. **Linear Regression:** trained on scaled features; checked residual plots, Q-Q plot, coefficients and VIF (multicollinearity)
7. **Decision Tree:** trained an untuned tree to show overfitting, then tuned `criterion`, `max_depth`, `min_samples_split` and `min_samples_leaf` with `GridSearchCV` (5-fold stratified CV, scored on F1)

## Results

All numbers below are on the held-out test set (8,300 orders).

### Linear Regression (delivery time in minutes)

| Metric | Train | Test |
|---|---|---|
| MAE (min) | 4.849 | 4.840 |
| RMSE (min) | 6.072 | 6.033 |
| R² | 0.583 | 0.580 |

The near-identical train and test scores show no overfitting. The model explains about 58% of the variation in delivery time, and predictions are off by roughly 5 minutes on average.

### Decision Tree classifier (On-time / Delayed)

| Metric | Test |
|---|---|
| Accuracy | 0.8551 |
| Precision (Delayed) | 0.8757 |
| Recall (Delayed) | 0.7949 |
| F1-score (Delayed) | 0.8333 |
| ROC-AUC | 0.9315 |

Train accuracy is 0.8645 against a test accuracy of 0.8551, a gap of about 1 point, so tuning controlled overfitting.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| On-time | 0.84 | 0.91 | 0.87 | 4517 |
| Delayed | 0.88 | 0.79 | 0.83 | 3783 |

The tree misses roughly 1 in 5 truly delayed orders (Delayed recall 0.79), which is its main limitation.

## Tools

Python, Pandas, NumPy, Scikit-learn, Statsmodels, Matplotlib, Seaborn, Google Colab

## Acknowledgements

Dataset by Gaurav Malik on Kaggle.
