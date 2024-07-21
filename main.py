import asyncio
import websockets
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from collections import deque
import os
from datetime import datetime
import pickle
import csv
import logging

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

logging.basicConfig(level=logging.DEBUG)

async def get_ticks_history():
    async with websockets.connect(connection_url) as websocket:
        await websocket.send(json.dumps(ticks_history_request))
        response = await websocket.recv()
        data = json.loads(response)
        if 'error' in data:
            logging.error(f"Error: {data['error']['message']}")
        else:
            return data['history']['prices']

async def subscribe_ticks():
    async with websockets.connect(connection_url) as websocket:
        await websocket.send(json.dumps(ticks_request))
        while True:
            response = await websocket.recv()
            data = json.loads(response)
            if 'error' in data:
                logging.error(f"Error: {data['error']['message']}")
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
    for i in range(65, len(prices_scaled)):
        X_train.append(prices_scaled[:i, 0])
        y_train.append(prices_scaled[i, 0])
    X_train = tf.keras.preprocessing.sequence.pad_sequences(
        X_train, maxlen=60, dtype='float32')
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    return X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train, scaler

def save_to_csv(data, filename='tick_data.csv'):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['timestamp', 'price'])
        writer.writerow(data)

def load_csv_data(filename='tick_data.csv'):
    df = pd.read_csv(filename)
    return df['price'].values.reshape(-1, 1)

def offline_prediction(model, scaler, data):
    window_size = 60
    predictions = []
    for i in range(len(data) - window_size - 5):
        input_data = data[:i+window_size]
        input_scaled = scaler.transform(input_data)
        input_reshaped = tf.keras.preprocessing.sequence.pad_sequences(
            [input_scaled], maxlen=60, dtype='float32')
        prediction_scaled = model.predict(input_reshaped)
        prediction = scaler.inverse_transform(prediction_scaled)[0][0]
        actual_price = data[i+window_size+5][0]
        predictions.append((prediction, actual_price))
    return predictions

def save_model(model, scaler, filename_prefix='lstm_model'):
    model_filename = f"{filename_prefix}.h5"
    model.save(model_filename)
    logging.info(f"Model saved as {model_filename}")

    scaler_filename = f"{filename_prefix}_scaler.pkl"
    with open(scaler_filename, 'wb') as f:
        pickle.dump(scaler, f)
    logging.info(f"Scaler saved as {scaler_filename}")

def load_latest_model(filename_prefix='lstm_model'):
    model_filename = f"{filename_prefix}.h5"
    scaler_filename = f"{filename_prefix}_scaler.pkl"

    if os.path.exists(model_filename) and os.path.exists(scaler_filename):
        model = tf.keras.models.load_model(model_filename)
        logging.info(f"Loaded model from {model_filename}")
        with open(scaler_filename, 'rb') as f:
            scaler = pickle.load(f)
        logging.info(f"Loaded scaler from {scaler_filename}")
        return model, scaler
    else:
        return None, None

def calculate_accuracy(predictions, actuals, threshold=0.0001):
    correct = sum(abs(pred - act) <= threshold for pred, act in zip(predictions, actuals))
    return correct / len(predictions) if predictions else 0

async def main():
    model, scaler = load_latest_model()

    if model is None or scaler is None:
        historical_prices = await get_ticks_history()
        historical_prices = np.array(
            [float(price) for price in historical_prices]).reshape(-1, 1)

        X_train, y_train, scaler = preprocess_data(historical_prices)
        model = create_lstm_model((X_train.shape[1], 1))
        model.fit(X_train, y_train, epochs=20, batch_size=32)
        save_model(model, scaler)
    else:
        model.compile(optimizer='adam', loss='mean_squared_error')
        historical_prices = await get_ticks_history()
        historical_prices = np.array(
            [float(price) for price in historical_prices]).reshape(-1, 1)

    window_size = 1000
    retrain_interval = 100
    tick_count = 0
    save_interval = 1000

    prices = deque([float(price)
                   for price in historical_prices[-window_size:, 0]], maxlen=window_size)
    predictions = deque(maxlen=100)  # Store last 100 predictions
    actuals = deque(maxlen=100)  # Store last 100 actual prices

    while True:
        try:
            async for tick in subscribe_ticks():
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                latest_price = float(tick['quote'])

                # Save tick data to CSV
                save_to_csv([timestamp, latest_price])

                # Make prediction for the next tick
                input_data = np.array(list(prices)).reshape(-1, 1)
                input_scaled = scaler.transform(input_data)
                input_reshaped = tf.keras.preprocessing.sequence.pad_sequences(
                    [input_scaled], maxlen=60, dtype='float32')

                logging.debug(f"Input reshaped for prediction: {input_reshaped.shape}")

                prediction_scaled = model.predict(input_reshaped)
                prediction = scaler.inverse_transform(prediction_scaled)[0][0]

                logging.debug(f"Current price: {latest_price}")
                logging.debug(f"Predicted price for next tick: {prediction}")

                # Store prediction and actual price
                predictions.append(prediction)
                actuals.append(latest_price)

                # Calculate and report accuracy
                if len(predictions) >= 10:  # Start reporting after 10 predictions
                    accuracy = calculate_accuracy(
                        list(predictions)[:-1], list(actuals)[1:])
                    logging.info(f"Current accuracy (last {len(predictions)-1} predictions): {accuracy:.2%}")

                    # Conditional retraining based on accuracy
                    if accuracy >= 0.8:  # If accuracy is 80% or more
                        retrain_interval = 50  # Retrain more frequently
                    else:
                        retrain_interval = 100  # Default retrain interval

                prices.append(latest_price)

                tick_count += 1
                if tick_count % retrain_interval == 0:
                    X_train, y_train, scaler = preprocess_data(
                        np.array(list(prices)).reshape(-1, 1))
                    # Recreate the model with the same architecture
                    model = create_lstm_model((X_train.shape[1], 1))
                    model.fit(X_train, y_train, epochs=10, batch_size=32)

                if tick_count % save_interval == 0:
                    save_model(model, scaler)

        except (websockets.ConnectionClosed, ConnectionError) as e:
            logging.error(f"Connection error: {e}. Switching to offline mode.")

            # Perform offline prediction
            historical_prices = load_csv_data()
            if len(historical_prices) > 60:
                X_train, _, scaler = preprocess_data(historical_prices)
                predictions = offline_prediction(model, scaler, historical_prices)

                for prediction, actual in predictions:
                    logging.info(f"Offline prediction: {prediction}, Actual price: {actual}")

            # Wait before trying to reconnect
            await asyncio.sleep(10)  # Wait for 10 seconds before retrying

if __name__ == "__main__":
    asyncio.run(main())
