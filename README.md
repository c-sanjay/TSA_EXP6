# Ex.No: 6               HOLT WINTERS METHOD
### Date: 20/05/2026
### NAME : SANJAY C
### REG NO : 212223240150
### AIM:
To implement the Holt Winters Method Model using Python.
### REQUIREMENTS:
```
1.DATASET : APPLE STOCK PRICE
2.TECHNOLOGY USED : GOOGLE COLLAB
```
### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv('apple.csv')

data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

data_time_series = data['Volume']

data_time_series.head()

data_yearly = data_time_series

data_yearly = data_yearly.asfreq('D')

data_yearly = data_yearly.ffill().bfill()

if data_yearly.min() <= 0:
    data_yearly = data_yearly + abs(data_yearly.min()) + 0.001

data_yearly.head()

data_yearly.plot()

scaler = MinMaxScaler(feature_range=(0.1, 1))

scaled_data = pd.Series(
    scaler.fit_transform(
        data_yearly.values.reshape(-1, 1)
    ).flatten(),
    index=data_yearly.index
)

full_data_scaler = MinMaxScaler(feature_range=(0.1, 1))
full_scaled_data = pd.Series(
    full_data_scaler.fit_transform(
        data_yearly.values.reshape(-1, 1)
    ).flatten(),
    index=data_yearly.index
)


scaled_data.plot()


from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(
    data_yearly.dropna(),
    model="additive",
    period=7
)

decomposition.plot()

plt.show()


scaled_data = scaled_data.ffill()

train_data = scaled_data[:int(len(scaled_data) * 0.8)]

test_data = scaled_data[int(len(scaled_data) * 0.8):]

model_add = ExponentialSmoothing(
    train_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=7
).fit()

test_predictions_add = model_add.forecast(steps=len(test_data))

ax = train_data.plot()

test_predictions_add.plot(ax=ax)

test_data.plot(ax=ax)

ax.legend(
    [
        "train_data",
        "test_predictions_add",
        "test_data"
    ]
)

ax.set_title('Visual evaluation')

plt.show()

print("RMSE:")
print(np.sqrt(mean_squared_error(test_data, test_predictions_add)))

print("Standard deviation and mean:")
print(np.sqrt(scaled_data.var()), scaled_data.mean())

final_model = ExponentialSmoothing(
    full_scaled_data,
    trend='add',
    seasonal='mul',
    seasonal_periods=7
).fit()

final_predictions_scaled = final_model.forecast(
    steps=int(len(data_yearly)/4)
)

final_predictions = pd.Series(
    full_data_scaler.inverse_transform(
        final_predictions_scaled.values.reshape(-1, 1)
    ).flatten(),
    index=final_predictions_scaled.index
)

ax = data_yearly.plot()

final_predictions.plot(ax=ax)

ax.legend(
    [
        "data_yearly",
        "final_predictions"
    ]
)

ax.set_xlabel('Date')

ax.set_ylabel('Volume')

ax.set_title('Prediction')

plt.show()
```
### OUTPUT:


#### TEST_PREDICTION

<img width="738" height="499" alt="image" src="https://github.com/user-attachments/assets/d3acb2f0-c201-4382-b090-d5353ae45920" />


#### FINAL_PREDICTION

<img width="743" height="507" alt="image" src="https://github.com/user-attachments/assets/2e804ea8-a311-4bf9-8a1f-e72081eeb2ad" />

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
