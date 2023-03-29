import numpy as np
import pandas as pd
import datetime as dt
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
import yfinance as yf

scaler = MinMaxScaler(feature_range=(0, 1))
past_days_used = 60
#future_days = 5

def train_transform(dataset):
    return np.reshape(dataset, (dataset.shape[0], dataset.shape[1], 1))

def train_test_data_creation(scaled_data):
    X, y = [], []
    for index in range(past_days_used, len(scaled_data)):
        X.append(scaled_data[index-past_days_used:index, 0])
        y.append(scaled_data[index, 0])
    X, y = np.array(X), np.array(y)
    X = train_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    #print(x_train.shape[1])
    return X_train, y_train, X_test, y_test

def data_preprocessing(value):
    df = yf.download(value, start=dt.date.today()-dt.timedelta(days=200), end=dt.date.today())
    dataframe = df['Close']
    scaled_data = scaler.fit_transform(dataframe.values.reshape(-1,1))
    #print(scaled_data.shape)
    return scaled_data

def LSTM_model():
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape = (past_days_used, 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))

    return model

def train_model(x_train, y_train):
    model = LSTM_model()
    model.compile(optimizer='adam', loss='mean_squared_error')
    checkpointer = ModelCheckpoint(filepath='weights_best.hd5', verbose=2, save_best_only=True)
    earlystopping = EarlyStopping(monitor='loss', patience=10)
    model.fit(x_train, y_train, epochs = 25, batch_size = 8,
    callbacks=[checkpointer, earlystopping], validation_split=0.1, shuffle=True)
    return model

def test_model(model, X_test, y_test):
    pass

def forecast_model_predict(model, input_data, future_days):
    input_data = train_transform(np.array([input_data[-60:,0]]))
    #print(input_data)
    prediction_array = []
    for _ in range(future_days):
        prediction = model.predict(input_data)
        prediction = scaler.inverse_transform(prediction)
        prediction_array.append(prediction[0][0]) #array in form [[number]]
        #print(prediction.shape)
        #print(train_transform(prediction).shape)
        input_data = np.concatenate([input_data, train_transform(prediction)], axis=1)[:,-60:,:]
    #print(input_data.shape)
    #print(input_data)
    return prediction_array

def main(value, future_days, use):
    scaled_db = data_preprocessing(value)
    if use == 'train':
        X_train, y_train, X_test, y_test = train_test_data_creation(scaled_db)
        #print(x_train[0].shape, x_train[0])
        model = train_model(X_train, y_train)
    #results = test_model(model, X_test, y_test)
        print("\nMode Trained")
    else:
        model = load_model('weights_best.hd5')
        prediction = forecast_model_predict(model, scaled_db, future_days)
        return prediction

# if __name__ == "__main__":
#     print(main("AAPL", 10, "train"))

