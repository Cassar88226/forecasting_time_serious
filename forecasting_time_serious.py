import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose 
from pmdarima.arima import auto_arima
plt.style.use('Solarize_Light2')
xl = pd.ExcelFile("RawData.xlsx")

df = xl.parse("RawData")
# print(df.head())
# df.Date = pd.to_datetime(df.index)
df['Power'] = df.Power
# df.Power.plot(figsize=(11,5),subplots = True);
# df.Temp.plot(figsize=(11,5),subplots = True);
# plt.title("Power and Temp Data");
# plt.show()


from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df.Power, model='multiplicative', freq=30)
result.plot()
plt.show()
data = df.Power
stepwise_model = auto_arima(data, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(stepwise_model.aic())
train = data.loc[:-24]
test = data.loc[-24:]
stepwise_model.fit(train)
# train_data = df[:len(df)-12]
# test_data = df[len(df)-12:]