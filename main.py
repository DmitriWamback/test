import yfinance

from datetime import date
import matplotlib.pyplot as plt

from prophet        import Prophet
from prophet.plot   import plot_plotly

import numpy as np
np.float = float

##########################################################################################

ticker = 'NVDA'
period_years = 2

start_date = '2020-01-01'
today_date = date.today().strftime('%Y-%m-%d')

stock = yfinance.download(ticker, start_date, today_date)
stock.reset_index(inplace=True)

##########################################################################################

df_train = stock[['Date', 'Close']]
df_train = df_train.rename(columns={'Date': 'ds', 'Close': 'y'})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=365 * period_years)
forecast = m.predict(future)

##########################################################################################

trends          = forecast['trend']
lower_estimate  = forecast['yhat_lower']
higher_estimate = forecast['yhat_upper']

highest = round(higher_estimate[len(higher_estimate)-1], 2)
lowest = round(lower_estimate[len(lower_estimate)-1], 2)

plt.title(f'{ticker} forecast')
plt.plot(trends,            color='black',  linewidth=2, label='Trend')
plt.plot(stock['Close'],    color='blue',   linewidth=1, label='Close Prices')
plt.plot(lower_estimate,    color='red',    linewidth=1, label=f'Lowest Estimate: {lowest}$')
plt.plot(higher_estimate,   color='green',  linewidth=1, label=f'Highest Estimate: {highest}$')
plt.legend(loc='upper right')
plt.show()

##########################################################################################