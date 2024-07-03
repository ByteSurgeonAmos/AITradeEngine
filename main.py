import asyncio
import websockets
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

app_id = 1089
connection_url = f'wss://ws.derivws.com/websockets/v3?app_id={app_id}'

ticks_history_request = {
    "ticks_history": "R_50",
    "adjust_start_time": 1,
    "count": 1000,
    "end": "latest",
    "start": 1,
    "style": "ticks",
}

ticks_request = {
    **ticks_history_request,
    "subscribe": 1,
}


async def get_ticks_history():
    async with websockets.connect(connection_url) as websocket:
        await websocket.send(json.dumps(ticks_history_request))
        response = await websocket.recv()
        data = json.loads(response)
        if 'error' in data:
            print(f"Error: {data['error']['message']}")
        else:
            return data['history']['prices']


async def subscribe_ticks():
    async with websockets.connect(connection_url) as websocket:
        await websocket.send(json.dumps(ticks_request))
        while True:
            response = await websocket.recv()
            data = json.loads(response)
            if 'error' in data:
                print(f"Error: {data['error']['message']}")
                break
            elif data['msg_type'] == 'tick':
                yield data['tick']


def create_lstm_model(input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(
        units=50, return_sequences=True, input_shape=input_shape))
    model.add(tf.keras.layers.LSTM(units=50))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def preprocess_data(prices):
    scaler = StandardScaler()
    prices_scaled = scaler.fit_transform(prices)
    X_train = []
    y_train = []
    for i in range(60, len(prices_scaled)):
        X_train.append(prices_scaled[i-60:i, 0])
        y_train.append(prices_scaled[i, 0])
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    return X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, scaler


async def main():
    prices = await get_ticks_history()
    prices = np.array(prices).reshape(-1, 1)

    # Preprocess the data
    X_train, y_train, scaler = preprocess_data(prices)

    # Create and train the LSTM model
    model = create_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=20, batch_size=32)

    window_size = 1000  # Define the window size for retraining
    retrain_interval = 100  # Define how often to retrain the model
    tick_count = 0

    # Subscribe to ticks and make predictions
    async for tick in subscribe_ticks():
        latest_price = np.array([[tick['quote']]])
        latest_price_scaled = scaler.transform(latest_price)

        # Update prices array and prepare the latest input for prediction
        prices = np.append(prices, latest_price).reshape(-1, 1)
        if len(prices) > window_size:
            prices = prices[-window_size:]

        latest_prices_scaled = scaler.transform(prices[-60:])
        latest_input = latest_prices_scaled.reshape(1, 60, 1)

        # Make prediction
        prediction_scaled = model.predict(latest_input)
        prediction = scaler.inverse_transform(prediction_scaled.reshape(-1, 1))

        print(f"Predicted price: {prediction[0][0]}")

        tick_count += 1
        if tick_count % retrain_interval == 0:
            # Retrain the model with the updated prices
            X_train, y_train, scaler = preprocess_data(prices)
            model = create_lstm_model((X_train.shape[1], 1))
            model.fit(X_train, y_train, epochs=10, batch_size=32)

if __name__ == "__main__":
    asyncio.run(main())
