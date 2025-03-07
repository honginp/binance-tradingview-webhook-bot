import json, config, jsonschema, logging
from flask import Flask, request
from api.binance_spot import BinanceSpotHttpClient
from api.binance_future import BinanceFutureHttpClient, OrderSide, OrderType
from event import EventEngine, Event, EVENT_TIMER, EVENT_SIGNAL
from decimal import Decimal

app = Flask(__name__)

# Load the JSON schema from a file (or define it as a Python dict)
with open("alert_schema.json", "r") as f:
    alert_json_schema = json.load(f)


@app.route('/', methods=['GET'])
def welcome():
    return "Hello Flask, This is for testing. If you receive this message, it means your configuration is correct."


@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = json.loads(request.data)

        # **JSON Schema Validation:**
        try:
            jsonschema.validate(instance=data, schema=alert_json_schema)
        except jsonschema.exceptions.ValidationError as e:
            print(f"JSON Schema Validation Error: {e}")
            return f"failure: JSON schema validation error - {e}", 400  # Return 400 Bad Request

        print(data)
        if data.get('passphrase', None) != config.WEBHOOK_PASSPHRASE:
            return "failure: passphrase is incorrect."

        event = Event(EVENT_SIGNAL, data=data)
        event_engine.put(event)

        return "success"

    except Exception as error:
        print(f"error: {error}")
        return "failure"

def check_payload(data: dict) -> tuple[bool, dict]:
    """
    check the payload from tradingview webhook.
    print and modify incomplete data following the schema.
    """
    strategy_config = config.strategies.get(data.get('strategy_name', ''), {})
    if not strategy_config:
        print('strategy_name cannot be found in config.')
        return False, data
    if data.get('price', '0') == '0':
        print('price cannot be 0 or None.')
        return False, data
    if data.get('action', '') == '':
        print('action cannot be None.')
        return False, data
    if data.get('symbol', '') == '':
        print('symbol cannot be None.')
        return False, data
    if strategy_config.get('trading_volume', 0) == 0 or strategy_config.get('trading_volume', 0) < strategy_config.get('min_volume', 0):
        print('trading_volume cannot be 0 or less than min_volume.')
        return False, data
    
    return True, data


def future_trade(data: dict):
    data = check_payload(data)
    symbol = data.get('symbol', None)
    action = data.get('action', '').upper()
    strategy_name = data.get('strategy_name', None)

    if not strategy_name:
        logging.error("strategy_name is None.")
        return

    strategy_config = config.strategies.get(strategy_name, None)
    if not strategy_config:
        logging.error("strategy_config is None.")
        return

    current_pos = strategy_config.get('pos', 0)
    trading_volume = strategy_config.get('trading_volume', 0)

    # print('strategy name', strategy_name, 'action: ', action, "current_pos: ", current_pos, "trading_volume: ", trading_volume)
    # print('order id :', future_strategy_order_dict.get(strategy_name))
    price = str(data.get('price', '0'))

    if action == 'EXIT':

        if current_pos > 0:

            vol1 = str(current_pos)

            order_id = binance_future_client.get_client_order_id()

            # the order support: LIMIT, MARKET, MAKER order
            # if you want to place a maker order, set the order_type=OrderType.MAKER
            # 支持限价单，市价单，做市单，如果你想下做市单，设置参数 order_type=OrderType.MAKER
            status, order = binance_future_client.place_order(
                symbol=symbol,
                order_side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal(vol1),
                price=Decimal(price),
                client_order_id=order_id
            )

            print("exit long: ", status, order)

            if status == 200:
                future_strategy_order_dict[strategy_name] = order_id

        elif current_pos < 0:

            vol1 = str(abs(current_pos))

            order_id = binance_future_client.get_client_order_id()
            status, order = binance_future_client.place_order(
                symbol=symbol,
                order_side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal(vol1),
                price=Decimal(price),
                client_order_id=order_id
            )
            print("exit short: ", status, order)
            if status == 200:
                future_strategy_order_dict[strategy_name] = order_id

    elif action == 'LONG':

        if current_pos < 0:

            vol1 = str(abs(current_pos) + trading_volume)
            order_id = binance_future_client.get_client_order_id()
            status, order = binance_future_client.place_order(
                symbol=symbol,
                order_side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal(vol1),
                price=Decimal(price),
                client_order_id=order_id
            )
            print("exit short & long: ", status, order)
            if status == 200:
                future_strategy_order_dict[strategy_name] = order_id

        if current_pos == 0:
            # config your trading volume in config.py

            vol1 = str(trading_volume)
            order_id = binance_future_client.get_client_order_id()
            status, order = binance_future_client.place_order(
                symbol=symbol,
                order_side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal(vol1),
                price=Decimal(price),
                client_order_id=order_id
            )
            print("long: ", status, order)
            if status == 200:
                future_strategy_order_dict[strategy_name] = order_id

    elif action == 'SHORT':

        if current_pos > 0:

            vol1 = str(abs(current_pos) + trading_volume)
            order_id = binance_future_client.get_client_order_id()
            status, order = binance_future_client.place_order(
                symbol=symbol,
                order_side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal(vol1),
                price=Decimal(price),
                client_order_id=order_id
            )
            print("exit long & short: ", status, order)
            if status == 200:
                future_strategy_order_dict[strategy_name] = order_id

        if current_pos == 0:
            vol1 = str(trading_volume)
            order_id = binance_future_client.get_client_order_id()
            status, order = binance_future_client.place_order(
                symbol=symbol,
                order_side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal(str(vol1)),
                price=Decimal(price),
                client_order_id=order_id
            )
            print("short: ", status, order)
            if status == 200:
                future_strategy_order_dict[strategy_name] = order_id


def timer_event(event: Event):
    global cancel_orders_timer
    global query_orders_timer

    cancel_orders_timer += 1
    query_orders_timer += 1

    if cancel_orders_timer > config.CANCEL_ORDERS_IN_SECONDS:  # for cancel order.
        cancel_orders_timer = 0 # reset the timer
        # will cancel the order repeatedly. the default value is CANCEL_ORDER_IN_SECONDS = 60
        for strategy_name in future_strategy_order_dict.keys():
            order_id = future_strategy_order_dict[strategy_name]
            if not order_id:
                continue

            symbol = config.strategies.get(strategy_name, {}).get('symbol', "")
            binance_future_client.cancel_order(symbol, client_order_id=order_id)

    if query_orders_timer > config.QUERY_ORDERS_STATUS_IN_SECONDS: # for updating order.
        query_orders_timer = 0 #  reset the query order timer.

        for strategy_name in future_strategy_order_dict.keys():
            order_id = future_strategy_order_dict[strategy_name]
            if not order_id:
                continue

            symbol = config.strategies.get(strategy_name, {}).get('symbol', "")

            status_code, order = binance_future_client.get_order(symbol, client_order_id=order_id)
            if status_code == 200 and order:
                if order.get('status') == 'CANCELED' or order.get('status') == 'FILLED':
                    side = order.get('side')
                    strategy_config = config.strategies.get(strategy_name, {})
                    executed_qty = Decimal(order.get('executedQty', "0"))

                    if side == "BUY":  # BUY
                        strategy_config['pos'] = strategy_config['pos'] + executed_qty

                    elif side == "SELL":  # SELL
                        strategy_config['pos'] = strategy_config['pos'] - executed_qty

                    # print(strategy_config)
                    config.strategies[strategy_name] = strategy_config  # update the data.
                    future_strategy_order_dict[strategy_name] = None
            elif status_code == 400 and order.get('code') == -2013:  # Order does not exist.
                future_strategy_order_dict[strategy_name] = None

    for strategy_name in future_signal_dict.keys():

        orderid = future_strategy_order_dict.get(strategy_name, None)
        if not orderid:
            data = future_signal_dict.get(strategy_name, None)
            if data:
                future_trade(data)

    for key in spot_signal_dict.keys():
        """
        check your spot signal here, whether your buy/sell order filled.
        
        please refer to the future signal and code your logic below.
        """
        pass


def signal_event(event: Event):
    """
    :param event: the event that contains the signal data
    the signal data like below.
    {'action': 'long',
    'symbol': 'ETHUSDT', 'exchange': 'binance_future',
    'price': '3054.66', 'close': '3054.66',
    'passphrase': 'your password'}

    :return: None
    """

    data = event.data
    strategy_name = data.get('strategy_name', None)
    logging.info(f"signal event: {data}")
    if not strategy_name:
        print("config from tradingview does not have strategy_name key.")
        return None

    if data.get('exchange', None) == 'binance_future':
        future_signal_dict[strategy_name] = data  # strategy_name -> data
        future_trade(data)

    elif data.get('exchange', None) == 'binance_spot':

        future_signal_dict[strategy_name] = data  # strategy_name -> data
        # write your logic code here.


if __name__ == '__main__':
    future_signal_dict = {}
    spot_signal_dict = {}

    future_strategy_order_dict = {}

    cancel_orders_timer = 0  # 撤单的timer.
    query_orders_timer = 0 # 查询订单的tismer.

    binance_spot_client = BinanceSpotHttpClient(api_key=config.API_KEY, secret=config.API_SECRET)
    binance_future_client = BinanceFutureHttpClient(api_key=config.API_KEY, secret=config.API_SECRET)

    event_engine = EventEngine(interval=1)  # you can update the loop interval.
    event_engine.start()
    event_engine.register(EVENT_TIMER, timer_event)
    event_engine.register(EVENT_SIGNAL, signal_event)

    app.run(host='127.0.0.1', port=8888, debug=False)
