import logging
import csv
import pickle
from datetime import datetime, timedelta
from collections import deque
import pandas as pd
import numpy as np
import json
import websockets
import asyncio
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU usage

# Configurable variables
CONFIG = {
    'stake': 10.0,  # Trade stake amount in USD
    'stop_loss_percent': 2.0,  # Stop-loss as percentage of entry price
    'stop_loss_absolute': 0.5,  # Absolute stop-loss in price units
    'take_profit_percent': 1.0,  # Take-profit as percentage of entry price
    'prediction_horizon': 10,  # Predict high/low over next 10 ticks
    'window_size': 60,  # Input window for LSTM
    'max_trades_per_hour': 120,  # One trade every 30 seconds
    'confidence_threshold': 0.03,  # Lowered for small price movements
    'retrain_interval': 50,  # Frequent retraining
    'save_interval': 1000,  # Ticks before saving model
    'monitoring_timeout': 300,  # Timeout for contract monitoring (5 minutes)
    'max_open_contracts': 5,  # Limit open contracts
    'websocket_timeout': 30,  # WebSocket keepalive timeout
}

# Deriv API setup
app_id = 85574  # Your App ID
api_token = "lUih4ezjsQwFivN"  # Your API Token
connection_url = f'wss://ws.derivws.com/websockets/v3?app_id={app_id}'

# Request templates
ticks_history_request = {
    "ticks_history": "R_50",
    "adjust_start_time": 1,
    "count": 5000,
    "end": "latest",
    "start": 1,
    "style": "ticks",
}

ticks_request = {
    **ticks_history_request,
    "subscribe": 1,
}

buy_request_template = {
    "buy": 1,
    "subscribe": 1,
    "price": CONFIG['stake'],
    "parameters": {
        "amount": CONFIG['stake'],
        "basis": "stake",
        "contract_type": "CALL",
        "currency": "USD",
        "duration": CONFIG['prediction_horizon'],
        "duration_unit": "t",
        "symbol": "R_50"
    }
}

open_contract_request = {
    "proposal_open_contract": 1,
    "subscribe": 1
}

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def authenticate(websocket):
    auth_request = {"authorize": api_token}
    await websocket.send(json.dumps(auth_request))
    response = await websocket.recv()
    data = json.loads(response)
    if 'error' in data:
        logging.error(f"Authentication error: {data['error']['message']}")
        raise Exception("Authentication failed")
    logging.info("Authenticated successfully")

async def get_ticks_history():
    async with websockets.connect(connection_url, ping_interval=20, ping_timeout=CONFIG['websocket_timeout']) as websocket:
        await authenticate(websocket)
        await websocket.send(json.dumps(ticks_history_request))
        response = await websocket.recv()
        data = json.loads(response)
        if 'error' in data:
            logging.error(f"Error fetching ticks: {data['error']['message']}")
            return None
        logging.info(f"Fetched {len(data['history']['prices'])} historical ticks")
        return np.array([float(price) for price in data['history']['prices']]).reshape(-1, 1)

async def subscribe_ticks():
    while True:
        try:
            async with websockets.connect(connection_url, ping_interval=20, ping_timeout=CONFIG['websocket_timeout']) as websocket:
                await authenticate(websocket)
                await websocket.send(json.dumps(ticks_request))
                while True:
                    response = await websocket.recv()
                    data = json.loads(response)
                    if 'msg_type' not in data:
                        logging.warning(f"Received message without msg_type: {data}")
                        continue
                    if 'error' in data:
                        logging.error(f"Error in subscription: {data['error']['message']}")
                        continue
                    if data['msg_type'] == 'tick':
                        logging.debug(f"Received tick: {data['tick']['quote']}")
                        yield data
                    elif data['msg_type'] == 'proposal_open_contract':
                        logging.debug(f"Received contract update: {data}")
                        yield data
                    else:
                        logging.debug(f"Unhandled message type: {data['msg_type']}")
        except (websockets.ConnectionClosed, Exception) as e:
            logging.error(f"WebSocket error: {e}. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)

async def subscribe_contracts():
    while True:
        try:
            async with websockets.connect(connection_url, ping_interval=20, ping_timeout=CONFIG['websocket_timeout']) as websocket:
                await authenticate(websocket)
                while True:
                    response = await websocket.recv()
                    data = json.loads(response)
                    if 'msg_type' not in data:
                        logging.warning(f"Contract subscription: Received message without msg_type: {data}")
                        continue
                    if 'error' in data:
                        logging.error(f"Contract subscription error: {data['error']['message']}")
                        continue
                    if data['msg_type'] == 'proposal_open_contract':
                        logging.debug(f"Contract subscription: Received update: {data}")
                        yield data
                    else:
                        logging.debug(f"Contract subscription: Unhandled message type: {data['msg_type']}")
        except (websockets.ConnectionClosed, Exception) as e:
            logging.error(f"Contract WebSocket error: {e}. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)

async def check_open_contracts(websocket, contract_id, entry_price, subscribed_contracts):
    if contract_id not in subscribed_contracts:
        request = open_contract_request.copy()
        request['contract_id'] = contract_id
        await websocket.send(json.dumps(request))
        subscribed_contracts.add(contract_id)
        logging.info(f"Subscribed to contract {contract_id}")
    
    start_time = datetime.now()
    async for data in subscribe_contracts():
        if (datetime.now() - start_time).total_seconds() > CONFIG['monitoring_timeout']:
            logging.warning(f"Timeout waiting for contract {contract_id} update")
            return None
        if data['msg_type'] == 'proposal_open_contract' and data.get('proposal_open_contract', {}).get('contract_id') == contract_id:
            logging.debug(f"Contract check response for {contract_id}: {data}")
            contract = data.get('proposal_open_contract', {})
            current_price = float(contract.get('current_spot', entry_price))
            return current_price
        elif 'error' in data:
            logging.error(f"Error checking contract {contract_id}: {data['error']['message']}")
            return None
    return None

async def close_contract(websocket, contract_id, subscribed_contracts):
    request = {"sell": str(contract_id), "price": 0}
    await websocket.send(json.dumps(request))
    response = await websocket.recv()
    data = json.loads(response)
    if 'error' in data:
        logging.error(f"Error closing contract {contract_id}: {data['error']['message']}")
        return False
    logging.info(f"Closed contract {contract_id}")
    subscribed_contracts.discard(contract_id)
    return True

async def close_stale_contracts(websocket, contract_ids):
    for contract_id in contract_ids:
        for _ in range(3):  # Retry up to 3 times
            success = await close_contract(websocket, contract_id, set())
            if success:
                logging.info(f"Successfully closed stale contract {contract_id}")
                break
            logging.warning(f"Retrying to close stale contract {contract_id}")
            await asyncio.sleep(1)
        else:
            logging.error(f"Failed to close stale contract {contract_id} after retries")

def calculate_technical_indicators(prices, window=14):
    df = pd.DataFrame(prices, columns=['price'])
    
    delta = df['price'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    rs = rs.replace([np.inf, -np.inf], np.nan)
    df['rsi'] = 100 - (100 / (1 + rs)).bfill().fillna(50)
    
    ema12 = df['price'].ewm(span=12, adjust=False).mean()
    ema26 = df['price'].ewm(span=26, adjust=False).mean()
    df['macd'] = (ema12 - ema26).fillna(0)
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean().fillna(0)
    
    df['sma20'] = df['price'].rolling(window=20).mean().bfill()
    df['std20'] = df['price'].rolling(window=20).std().bfill().fillna(0)
    df['upper_band'] = df['sma20'] + (df['std20'] * 2)
    df['lower_band'] = df['sma20'] - (df['std20'] * 2)
    
    high_low = df['price'].rolling(window=2).max() - df['price'].rolling(window=2).min()
    df['atr'] = high_low.rolling(window=window).mean().bfill().fillna(0)
    
    low_14 = df['price'].rolling(window=window).min()
    high_14 = df['price'].rolling(window=window).max()
    denominator = high_14 - low_14
    denominator = denominator.replace(0, np.nan)
    df['stochastic'] = 100 * (df['price'] - low_14) / denominator.bfill().fillna(50)
    
    df['cum_price'] = df['price'].cumsum()
    df['vwap'] = df['cum_price'] / (df.index + 1)
    
    features = df[['price', 'rsi', 'macd', 'upper_band', 'lower_band', 'atr', 'stochastic', 'vwap']].bfill()
    if features.isna().any().any():
        logging.warning(f"NaN detected in features: {features.isna().sum()}")
        features = features.fillna({'rsi': 50, 'stochastic': 50, 'atr': 0, 'std20': 0, 'upper_band': df['sma20'], 'lower_band': df['sma20']})
    logging.debug(f"Features shape: {features.shape}, NaN count: {features.isna().sum().sum()}")
    return features.values

def create_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(units=64, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(units=32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(2)  # Predict high and low
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def preprocess_data(prices, window_size=CONFIG['window_size'], prediction_horizon=CONFIG['prediction_horizon']):
    features = calculate_technical_indicators(prices)
    if np.isnan(features).any():
        logging.warning("NaN in features before scaling. Filling with defaults.")
        features = pd.DataFrame(features, columns=['price', 'rsi', 'macd', 'upper_band', 'lower_band', 'atr', 'stochastic', 'vwap'])
        features = features.fillna({'rsi': 50, 'stochastic': 50, 'atr': 0, 'upper_band': features['price'], 'lower_band': features['price']}).values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    X_train, y_train_high, y_train_low = [], [], []
    for i in range(window_size, len(features_scaled) - prediction_horizon):
        X_train.append(features_scaled[i-window_size:i])
        future_prices = prices[i:i+prediction_horizon, 0]
        if len(future_prices) < prediction_horizon:
            continue
        y_train_high.append(np.max(future_prices))
        y_train_low.append(np.min(future_prices))
    
    X_train = np.array(X_train)
    y_train = np.array([y_train_high, y_train_low]).T
    if np.isnan(y_train).any():
        logging.warning("NaN in y_train. Filling with mean.")
        y_train = np.nan_to_num(y_train, nan=np.nanmean(y_train))
    if len(X_train) < 10:
        logging.error(f"Insufficient training samples: {len(X_train)}. Need more data.")
        return None, None, None, None
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    
    logging.info(f"Preprocessed data: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
    return X_train, y_train_scaled, scaler, y_scaler

def save_to_csv(data, filename='tick_data.csv'):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(['timestamp', 'price'])
        writer.writerow(data)

def load_csv_data(filename='tick_data.csv'):
    if os.path.exists(filename):
        df = pd.read_csv(filename)
        return df['price'].values.reshape(-1, 1)
    return None

def save_model(model, scaler, y_scaler, filename_prefix='lstm_model'):
    model_filename = f"{filename_prefix}.keras"
    model.save(model_filename)
    logging.info(f"Model saved as {model_filename}")
    with open(f"{filename_prefix}_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    with open(f"{filename_prefix}_y_scaler.pkl", 'wb') as f:
        pickle.dump(y_scaler, f)
    logging.info(f"Scalers saved")

def load_latest_model(filename_prefix='lstm_model'):
    model_filename = f"{filename_prefix}.keras"
    scaler_filename = f"{filename_prefix}_scaler.pkl"
    y_scaler_filename = f"{filename_prefix}_y_scaler.pkl"
    
    if os.path.exists(model_filename) and os.path.exists(scaler_filename) and os.path.exists(y_scaler_filename):
        model = tf.keras.models.load_model(model_filename)
        with open(scaler_filename, 'rb') as f:
            scaler = pickle.load(f)
        with open(y_scaler_filename, 'rb') as f:
            y_scaler = pickle.load(f)
        logging.info(f"Loaded model and scalers")
        return model, scaler, y_scaler
    return None, None, None

async def place_trade(websocket, contract_type, open_contracts):
    if len(open_contracts) >= CONFIG['max_open_contracts']:
        logging.warning(f"Max open contracts ({CONFIG['max_open_contracts']}) reached. Skipping trade.")
        return None
    buy_request = buy_request_template.copy()
    buy_request['parameters']['contract_type'] = contract_type
    buy_request['parameters']['amount'] = CONFIG['stake']
    buy_request['price'] = CONFIG['stake']
    await websocket.send(json.dumps(buy_request))
    response = await websocket.recv()
    data = json.loads(response)
    if 'error' in data:
        logging.error(f"Trade error: {data['error']['message']}")
        return None
    contract_id = data.get('buy', {}).get('contract_id')
    entry_price = float(data.get('buy', {}).get('buy_price', CONFIG['stake']))
    logging.info(f"Placed {contract_type} trade, contract_id: {contract_id}, entry_price: {entry_price}")
    return contract_id, entry_price

async def monitor_trade(websocket, contract_id, entry_price, contract_type, subscribed_contracts):
    stop_loss_price = entry_price * (1 - CONFIG['stop_loss_percent'] / 100) if contract_type == "CALL" else entry_price * (1 + CONFIG['stop_loss_percent'] / 100)
    absolute_stop_loss = entry_price - CONFIG['stop_loss_absolute'] if contract_type == "CALL" else entry_price + CONFIG['stop_loss_absolute']
    stop_loss_price = min(stop_loss_price, absolute_stop_loss) if contract_type == "CALL" else max(stop_loss_price, absolute_stop_loss)
    take_profit_price = entry_price * (1 + CONFIG['take_profit_percent'] / 100) if contract_type == "CALL" else entry_price * (1 - CONFIG['take_profit_percent'] / 100)
    
    start_time = datetime.now()
    while (datetime.now() - start_time).total_seconds() < CONFIG['monitoring_timeout']:
        current_price = await check_open_contracts(websocket, contract_id, entry_price, subscribed_contracts)
        if current_price is None:
            logging.warning(f"Failed to retrieve current price for contract {contract_id}. Stopping monitoring.")
            break
        logging.debug(f"Monitoring contract {contract_id}: Current price {current_price}, Stop-loss {stop_loss_price}, Take-profit {take_profit_price}")
        if (contract_type == "CALL" and current_price <= stop_loss_price) or (contract_type == "PUT" and current_price >= stop_loss_price):
            logging.info(f"Stop-loss triggered for contract {contract_id}. Current price: {current_price}, Stop-loss: {stop_loss_price}")
            await close_contract(websocket, contract_id, subscribed_contracts)
            break
        if (contract_type == "CALL" and current_price >= take_profit_price) or (contract_type == "PUT" and current_price <= take_profit_price):
            logging.info(f"Take-profit triggered for contract {contract_id}. Current price: {current_price}, Take-profit: {take_profit_price}")
            await close_contract(websocket, contract_id, subscribed_contracts)
            break
        await asyncio.sleep(1)
    else:
        logging.warning(f"Monitoring timeout for contract {contract_id}. Attempting to close.")
        await close_contract(websocket, contract_id, subscribed_contracts)

def calculate_accuracy(predictions, actuals, threshold=0.3):
    valid_pairs = [(pred, act) for pred, act in zip(predictions, actuals) if not np.isnan(pred) and not np.isnan(act)]
    if not valid_pairs:
        logging.warning("No valid prediction-actual pairs for accuracy calculation")
        return 0
    errors = [abs(pred - act) for pred, act in valid_pairs]
    correct = sum(error <= threshold for error in errors)
    accuracy = correct / len(valid_pairs)
    logging.debug(f"Accuracy calculation: Errors {errors[:5]}..., Mean error {np.mean(errors):.4f}, Threshold {threshold}")
    return accuracy

async def main():
    model, scaler, y_scaler = load_latest_model()
    prices = deque(maxlen=CONFIG['window_size'] + 100)
    trade_count = 0
    last_trade_time = datetime.now() - timedelta(seconds=3600 / CONFIG['max_trades_per_hour'])
    tick_count = 0
    open_contracts = {}
    subscribed_contracts = set()

    # Initialize data
    historical_prices = await get_ticks_history()
    if historical_prices is None:
        logging.error("Failed to fetch historical data. Exiting.")
        return
    
    prices.extend(historical_prices[-CONFIG['window_size']-100:, 0])
    logging.info(f"Initialized with {len(prices)} historical prices")
    
    if model is None or scaler is None or y_scaler is None:
        X_train, y_train_scaled, scaler, y_scaler = preprocess_data(historical_prices)
        if X_train is None:
            logging.error("Failed to preprocess data. Exiting.")
            return
        model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train_scaled, epochs=20, batch_size=32, verbose=1)
        save_model(model, scaler, y_scaler)
    else:
        model.compile(optimizer='adam', loss='mean_squared_error')

    # Close stale contracts
    async with websockets.connect(connection_url, ping_interval=20, ping_timeout=CONFIG['websocket_timeout']) as websocket:
        await authenticate(websocket)
        await close_stale_contracts(websocket, [288188143008, 288188674508])

    predictions_high, predictions_low, actuals_high, actuals_low = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)

    async with websockets.connect(connection_url, ping_interval=20, ping_timeout=CONFIG['websocket_timeout']) as websocket:
        await authenticate(websocket)
        async for data in subscribe_ticks():
            try:
                if data['msg_type'] == 'tick':
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                    latest_price = float(data['tick']['quote'])
                    save_to_csv([timestamp, latest_price])
                    prices.append(latest_price)
                    tick_count += 1
                    logging.debug(f"Tick {tick_count}: Price {latest_price}")

                    # Calculate features
                    features = calculate_technical_indicators(np.array(list(prices)).reshape(-1, 1))
                    if len(features) < CONFIG['window_size']:
                        logging.warning(f"Insufficient features: {len(features)} < {CONFIG['window_size']}. Waiting for more data.")
                        continue
                    
                    input_scaled = scaler.transform(features[-CONFIG['window_size']:])
                    if np.isnan(input_scaled).any():
                        logging.warning(f"NaN in input_scaled: {np.isnan(input_scaled).sum()}")
                        continue
                    input_reshaped = np.array([input_scaled]).reshape(1, CONFIG['window_size'], -1)
                    
                    # Predict high and low
                    try:
                        prediction_scaled = model.predict(input_reshaped, verbose=0)
                        prediction_high, prediction_low = y_scaler.inverse_transform(prediction_scaled)[0]
                        if np.isnan(prediction_high) or np.isnan(prediction_low):
                            logging.warning("NaN in predictions. Retraining model.")
                            X_train, y_train_scaled, scaler, y_scaler = preprocess_data(np.array(list(prices)).reshape(-1, 1))
                            if X_train is None:
                                logging.warning("Insufficient data for retraining. Waiting for more ticks.")
                                continue
                            model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
                            model.fit(X_train, y_train_scaled, epochs=10, batch_size=32, verbose=1)
                            save_model(model, scaler, y_scaler)
                            continue
                        logging.info(f"Current price: {latest_price:.4f}, Predicted high: {prediction_high:.4f}, Predicted low: {prediction_low:.4f}")
                    except Exception as e:
                        logging.error(f"Prediction error: {e}")
                        continue
                    
                    # Store predictions and actuals
                    future_prices = np.array(list(prices)[-CONFIG['prediction_horizon']:]) if len(prices) >= CONFIG['prediction_horizon'] else np.array(list(prices))
                    actual_high, actual_low = np.max(future_prices), np.min(future_prices)
                    predictions_high.append(prediction_high)
                    predictions_low.append(prediction_low)
                    actuals_high.append(actual_high)
                    actuals_low.append(actual_low)
                    
                    # Calculate accuracy
                    if len(predictions_high) >= 10:
                        high_accuracy = calculate_accuracy(list(predictions_high)[:-1], list(actuals_high)[1:])
                        low_accuracy = calculate_accuracy(list(predictions_low)[:-1], list(actuals_low)[1:])
                        logging.info(f"High accuracy: {high_accuracy:.2%}, Low accuracy: {low_accuracy:.2%}")
                        price_diff = prediction_high - prediction_low
                        if high_accuracy < 0.5 or low_accuracy < 0.5 or price_diff < 0.05:
                            logging.warning(f"Accuracy below 50% or price_diff {price_diff:.4f} too low. Forcing retraining.")
                            X_train, y_train_scaled, scaler, y_scaler = preprocess_data(np.array(list(prices)).reshape(-1, 1))
                            if X_train is not None:
                                model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
                                model.fit(X_train, y_train_scaled, epochs=10, batch_size=32, verbose=1)
                                save_model(model, scaler, y_scaler)
                        CONFIG['retrain_interval'] = 50 if (high_accuracy >= 0.8 and low_accuracy >= 0.8) else 100

                    # Trading logic
                    price_diff = prediction_high - prediction_low
                    atr = features[-1, 5]
                    stochastic = features[-1, 6]
                    vwap = features[-1, 7]
                    
                    if np.isnan(price_diff):
                        logging.info(f"No trade: NaN price_diff due to invalid predictions")
                        continue
                    
                    time_since_last_trade = (datetime.now() - last_trade_time).total_seconds()
                    if time_since_last_trade < 3600 / CONFIG['max_trades_per_hour']:
                        wait_time = 3600 / CONFIG['max_trades_per_hour'] - time_since_last_trade
                        logging.info(f"No trade: Waiting {wait_time:.1f} seconds for next trade opportunity. Price diff: {price_diff:.4f}, ATR: {0.5 * atr:.4f}")
                        continue
                    
                    if price_diff > CONFIG['confidence_threshold']:
                        if latest_price < prediction_low + 0.5 * price_diff:
                            contract_type = "CALL"
                        elif latest_price > prediction_high - 0.5 * price_diff:
                            contract_type = "PUT"
                        else:
                            contract_type = None
                            logging.info(f"No trade: Price diff {price_diff:.4f}, ATR {atr:.4f}, Stochastic {stochastic:.2f}, VWAP {vwap:.4f}, Current price not in range")
                    else:
                        contract_type = None
                        logging.info(f"No trade: Insufficient price diff {price_diff:.4f}")
                    
                    if contract_type:
                        logging.info(f"Attempting {contract_type} trade. Price diff: {price_diff:.4f}, ATR: {atr:.4f}, Stochastic: {stochastic:.2f}, VWAP: {vwap:.4f}")
                        result = await place_trade(websocket, contract_type, open_contracts)
                        if result:
                            contract_id, entry_price = result
                            trade_count += 1
                            last_trade_time = datetime.now()
                            CONFIG['stop_loss_absolute'] = max(CONFIG['stop_loss_absolute'], 0.5 * atr)
                            open_contracts[contract_id] = (entry_price, contract_type)
                            asyncio.create_task(monitor_trade(websocket, contract_id, entry_price, contract_type, subscribed_contracts))
                            logging.info(f"Trade count: {trade_count}")

                    # Retrain model
                    if tick_count % CONFIG['retrain_interval'] == 0:
                        X_train, y_train_scaled, scaler, y_scaler = preprocess_data(np.array(list(prices)).reshape(-1, 1))
                        if X_train is None:
                            logging.warning("Insufficient data for retraining. Waiting for more ticks.")
                            continue
                        model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
                        model.fit(X_train, y_train_scaled, epochs=10, batch_size=32, verbose=1)
                        save_model(model, scaler, y_scaler)

                    if tick_count % CONFIG['save_interval'] == 0:
                        save_model(model, scaler, y_scaler)

            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())