import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import yfinance as fn

st.title('Stock Price Prediction of Apple(AAPL)')

st.balloons()
#Describing the data

st.subheader(' Raw Data From 2018 - 2020')

df = pd.read_csv("AAPL.csv")
fd = pd.read_csv("AAPL.csv")
st.balloons()
st.write(df.describe())

def get_ticker(name):
    comp = fn.Ticker(name)
    return comp



st.write(""" ### About Apple  """)
c1 = get_ticker("AAPL")
st.write(c1.info['longBusinessSummary'])
st.balloons()
#Chart

st.subheader("Open Price  Graph")
st.line_chart(data = fd, y="Open")
st.balloons()


df = df.iloc[:, 1:2]
#Training the dataset
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range = (0, 1))

df_scaled = scaler.fit_transform(df)

forecast_features_set = []
labels = []
for i in range(60, 614):
    forecast_features_set.append(df_scaled[i-60:i, 0])
    labels.append(df_scaled[i, 0])

forecast_features_set , labels = np.array(forecast_features_set ), np.array(labels)
forecast_features_set = np.reshape(forecast_features_set, (forecast_features_set.shape[0], forecast_features_set.shape[1], 1))
#forecast_features_set.shape 


#Loading the model
model = load_model('Keras_model2.h5')

#Forecasting
forecast_list=[]

batch=df_scaled[-forecast_features_set.shape[1]:].reshape((1,forecast_features_set.shape[1],1))

for i in range(forecast_features_set.shape[1]):
    forecast_list.append(model.predict(batch)[0])
    batch = np.append(batch[:,1:,:], [[forecast_list[i]]], axis=1)


#Making Predicitons

df_predict=pd.DataFrame(scaler.inverse_transform(forecast_list),index=df[-forecast_features_set.shape[1]:].index, 
columns=["prediction"])

df_predict =pd.concat([df,df_predict],axis=1)
#st.write(df_predict.tail(10))


add_dates = [df.index[-1]+ x for x in range(0, 61)]
future_dates = pd.DataFrame(index = add_dates[1:], columns=df.columns)

df_forecast=pd.DataFrame(scaler.inverse_transform(forecast_list), columns=["prediction"],index=future_dates[-forecast_features_set.shape[1]:].index)            

df_forecast =pd.concat([df,df_forecast],axis=1)
st.subheader('60 days Predicted values ')
st.write(df_forecast.prediction.tail(60))
st.balloons()


st.subheader('Forecasted graph')
fig = plt.figure(figsize=(12,8))

plt.plot(df_predict["Open"],color="r",label="Actual Value")
plt.plot(df_forecast["prediction"],label="Forecasted Value")
plt.legend(loc='best',fontsize='large')
plt.xlabel("Index")
plt.ylabel("Open stock Price")
st.pyplot(fig)
plt.legend()
st.balloons()

