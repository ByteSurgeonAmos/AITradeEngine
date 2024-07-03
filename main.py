import asyncio
import websockets
import json
from sklearn.linear_model import LinearRegression
import numpy as np


app_id = 1089
connection_url = f'wss://ws.derivws.com/websockets/v3?app_id={app_id}'

ticks_history_request = {
    "ticks_history": "R_50",
    "adjust_start_time": 1,
    "count": 10,
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
        await websocket.send(json.dumps({"ticks_history": "R_50", "adjust_start_time": 1, "count": 1000, "end": "latest", "start": 1, "style": "ticks"}))
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
                # print(data)
                yield data['tick']


def create_model():
    model = LinearRegression()
    return model


async def main():
    prices = await get_ticks_history()
    prices = np.array(prices).reshape(-1, 1)

    # Create the dataset
    X_train = []
    y_train = []
    for i in range(60, len(prices)):
        X_train.append(prices[i-60:i, 0])
        y_train.append(prices[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)

    # Create and train the model
    model = create_model()
    model.fit(X_train, y_train)

    # Subscribe to ticks and make predictions
    async for tick in subscribe_ticks():
        latest_prices = np.append(prices, tick['quote']).reshape(-1, 1)
        latest_input = latest_prices[-60:].reshape(1, -1)
        prediction = model.predict(latest_input)
        # print(f"Predicted price: {prediction[0]}")

if __name__ == "__main__":
    asyncio.run(main())
