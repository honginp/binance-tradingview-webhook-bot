import json
import config
import jsonschema
import logging
from flask import Flask, request
from api.binance_spot import BinanceSpotHttpClient
from api.binance_future import BinanceFutureHttpClient, OrderSide, OrderType
from event import EventEngine, Event, EVENT_TIMER, EVENT_SIGNAL
from decimal import Decimal

app = Flask(__name__)

with open("alert_schema.json", "r") as f:
    alert_json_schema = json.load(f)

@app.route('/', methods=['GET'])
def welcome():
    return "Hello Flask, This is for testing."

@app.route('/webhook', methods=['POST'])
def webhook():
    try:
        data = json.loads(request.data)
        jsonschema.validate(instance=data, schema=alert_json_schema)
        if data.get('passphrase') != config.WEBHOOK_PASSPHRASE:
            return "failure: passphrase is incorrect.", 401
        event = Event(EVENT_SIGNAL, data=data)
        event_engine.put(event)
        return "success"
    except jsonschema.exceptions.ValidationError as e:
        logging.error(f"JSON Schema Validation Error: {e}")
        return f"failure: JSON schema validation error - {e}", 400
    except Exception as e:
        logging.error(f"Webhook error: {e}")
        return "failure", 500

def check_payload(data: dict, strategy_config: dict) -> tuple[bool, str]:
    """Validate payload strictly with no defaults."""
    strategy_name = data.get('strategy_name', '').strip()
    if not strategy_name or strategy_name not in config.strategies:
        return False, f"strategy_name '{strategy_name}' not found in config."

    symbol = data.get('symbol', '').strip()
    if not symbol:
        return False, "symbol cannot be None."
    symbol = symbol.split(':')[1] if ':' in symbol else symbol
    if symbol not in strategy_config.get('symbol', []):
        return False, f"Symbol {symbol} not allowed for strategy {strategy_name}."

    orders = data.get('order', [])
    if not orders:
        return False, "order array cannot be empty."

    order_types = {'market': OrderType.MARKET, 'limit': OrderType.LIMIT, 'stop': OrderType.STOP, 'maker': OrderType.MAKER}
    entry_ids = set()
    for order in orders:
        required = ['position_type', 'action', 'order_type', 'price', 'qty_type', 'qty', 'order_id']
        if not all(k in order for k in required):
            return False, f"Missing required field in order: {order}"
        if order['position_type'] not in ['entry', 'exit']:
            return False, f"Invalid position_type '{order['position_type']}'"
        if order['action'] not in ['long', 'short']:
            return False, f"Invalid action '{order['action']}'"
        if order['order_type'] not in order_types:
            return False, f"Invalid order_type '{order['order_type']}'"
        if order['position_type'] == 'exit' and 'exit_type' not in order:
            return False, f"exit_type missing for exit order: {order}"
        if order['position_type'] == 'exit' and order['exit_type'] not in ['take_profit', 'stop_loss']:
            return False, f"Invalid exit_type '{order['exit_type']}'"
        if order['qty_type'] not in ['percent', 'absolute']:
            return False, f"Invalid qty_type '{order['qty_type']}'"
        try:
            Decimal(order['price'])
            Decimal(order['qty'])
        except (ValueError, TypeError):
            return False, f"Invalid price or qty in order: {order}"
        if order['position_type'] == 'entry':
            entry_ids.add(order['order_id'])
        elif 'entry_order_id' in order and order['entry_order_id'] not in entry_ids and order['entry_order_id'] != 'ALL':
            return False, f"exit order references unknown entry_order_id '{order['entry_order_id']}'"

    trading_volume = strategy_config.get('trading_volume', Decimal('0'))
    min_volume = strategy_config.get('min_volume', Decimal('0'))
    if trading_volume <= Decimal('0') or trading_volume < min_volume:
        return False, f"trading_volume {trading_volume} must be > 0 and >= min_volume {min_volume}."

    return True, ""

def get_position_info(symbol: str) -> tuple[Decimal, Decimal]:
    """Fetch position info efficiently."""
    try:
        positions = binance_future_client.get_position_information(symbol=symbol)
        for position in positions:
            if position['symbol'] == symbol:
                qty = Decimal(position['positionAmt'])
                price = Decimal(position['entryPrice']) if qty != Decimal('0') else Decimal('0')
                return price, qty
        return Decimal('0'), Decimal('0')
    except Exception as e:
        logging.error(f"Error fetching position info for {symbol}: {e}")
        return Decimal('0'), Decimal('0')

def future_trade(data: dict):
    strategy_name = data.get('strategy_name')
    if not strategy_name or strategy_name not in config.strategies:
        logging.error(f"Invalid or missing strategy_name: {strategy_name}")
        return

    strategy_config = config.strategies[strategy_name]
    is_valid, error_msg = check_payload(data, strategy_config)
    if not is_valid:
        logging.error(f"Payload validation failed: {error_msg}")
        return

    symbol = data['symbol'].split(':')[1] if ':' in data['symbol'] else data['symbol']
    orders = data['order']
    avg_price, current_pos = get_position_info(symbol)
    trading_volume = strategy_config['trading_volume']
    executed_orders = []  # Track orders for rollback

    try:
        for order in orders:
            position_type = order['position_type']
            action = order['action']
            order_type = getattr(OrderType, order['order_type'].upper())
            price = Decimal(order['price'])
            qty_type = order['qty_type']
            qty = Decimal(order['qty'])
            order_id = order['order_id']
            order_side = OrderSide.BUY if action == 'long' else OrderSide.SELL

            if position_type == 'entry':
                entry_qty = trading_volume if qty_type == 'absolute' else trading_volume * (qty / Decimal('100'))
                if entry_qty < strategy_config['min_volume']:
                    raise ValueError(f"Entry qty {entry_qty} below min_volume {strategy_config['min_volume']}.")

                status, entry_order = binance_future_client.place_order(
                    symbol=symbol,
                    order_side=order_side,
                    order_type=order_type,
                    quantity=entry_qty,
                    price=price if order_type in [OrderType.LIMIT, OrderType.STOP] else None,
                    client_order_id=order_id
                )
                if status != 200:
                    raise RuntimeError(f"Entry order failed: {status}, {entry_order}")
                executed_orders.append((symbol, order_id))
                future_strategy_order_dict[f"{strategy_name}_{order_id}"] = order_id

            elif position_type == 'exit':
                exit_type = order['exit_type']
                entry_order_id = order.get('entry_order_id', 'ALL')  # Default to apply to entire position
                is_long = current_pos > 0
                position_size = abs(current_pos) if (is_long and action == 'long') or (not is_long and action == 'short') else Decimal('0')

                if position_size == Decimal('0'):
                    raise ValueError(f"No matching position for {action} to apply {exit_type}.")

                exit_qty = qty if qty_type == 'absolute' else position_size * (qty / Decimal('100'))
                if exit_qty > position_size:
                    raise ValueError(f"Exit qty {exit_qty} exceeds position size {position_size} for {exit_type}.")

                price_valid = True
                if order_type in [OrderType.LIMIT, OrderType.STOP]:
                    if exit_type == 'take_profit':
                        price_valid = (action == 'long' and price > avg_price) or (action == 'short' and price < avg_price)
                    elif exit_type == 'stop_loss':
                        price_valid = (action == 'long' and price < avg_price) or (action == 'short' and price > avg_price)
                    if not price_valid:
                        raise ValueError(f"Invalid {exit_type} price {price} for {action} position at {avg_price}.")

                exit_side = OrderSide.SELL if action == 'long' else OrderSide.BUY
                status, exit_order = binance_future_client.place_order(
                    symbol=symbol,
                    order_side=exit_side,
                    order_type=order_type,
                    quantity=exit_qty,
                    price=price if order_type in [OrderType.LIMIT, OrderType.STOP] else None,
                    client_order_id=order_id,
                    reduceOnly=(entry_order_id != 'ALL')  # ReduceOnly for specific entry linkage
                )
                if status != 200:
                    raise RuntimeError(f"Exit order failed: {status}, {exit_order}")
                executed_orders.append((symbol, order_id))
                future_strategy_order_dict[f"{strategy_name}_{order_id}"] = order_id

    except Exception as e:
        logging.error(f"Processing failed: {e}. Rolling back executed orders: {executed_orders}")
        for sym, oid in executed_orders:
            binance_future_client.cancel_order(sym, client_order_id=oid)
            future_strategy_order_dict.pop(f"{strategy_name}_{oid}", None)
        return

def timer_event(event: Event):
    global cancel_orders_timer, query_orders_timer
    cancel_orders_timer += 1
    query_orders_timer += 1

    if cancel_orders_timer > config.CANCEL_ORDERS_IN_SECONDS:
        cancel_orders_timer = 0
        for key in list(future_strategy_order_dict.keys()):
            order_id = future_strategy_order_dict[key]
            if order_id:
                strategy_name = key.split('_')[0]
                symbol = config.strategies.get(strategy_name, {}).get('symbol', [''])[0]
                if symbol:
                    binance_future_client.cancel_order(symbol, client_order_id=order_id)
                    future_strategy_order_dict[key] = None

    if query_orders_timer > config.QUERY_ORDERS_STATUS_IN_SECONDS:
        query_orders_timer = 0
        for key in list(future_strategy_order_dict.keys()):
            order_id = future_strategy_order_dict[key]
            if order_id:
                strategy_name = key.split('_')[0]
                symbol = config.strategies.get(strategy_name, {}).get('symbol', [''])[0]
                if symbol:
                    status_code, order = binance_future_client.get_order(symbol, client_order_id=order_id)
                    if status_code == 200 and order and order.get('status') in ['CANCELED', 'FILLED']:
                        strategy_config = config.strategies[strategy_name]
                        qty = Decimal(order.get('executedQty', '0'))
                        if order.get('side') == 'BUY':
                            strategy_config['pos'] += qty
                        elif order.get('side') == 'SELL':
                            strategy_config['pos'] -= qty
                        config.strategies[strategy_name] = strategy_config
                        future_strategy_order_dict[key] = None

def signal_event(event: Event):
    data = event.data
    strategy_name = data.get('strategy_name')
    if not strategy_name:
        logging.error("No strategy_name in signal.")
        return
    if data.get('exchange') == 'binance_future':
        future_signal_dict[strategy_name] = data
        future_trade(data)

if __name__ == '__main__':
    future_signal_dict = {}
    spot_signal_dict = {}
    future_strategy_order_dict = {}
    cancel_orders_timer = 0
    query_orders_timer = 0

    binance_spot_client = BinanceSpotHttpClient(api_key=config.API_KEY, secret=config.API_SECRET)
    binance_future_client = BinanceFutureHttpClient(api_key=config.API_KEY, secret=config.API_SECRET)

    event_engine = EventEngine(interval=1)
    event_engine.start()
    event_engine.register(EVENT_TIMER, timer_event)
    event_engine.register(EVENT_SIGNAL, signal_event)

    app.run(host='127.0.0.1', port=8888, debug=False)