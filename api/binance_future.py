import requests
import time
import hmac
import hashlib
from threading import Lock
from datetime import datetime
from decimal import Decimal
import json

from .constant import RequestMethod, Interval, OrderSide, OrderType


class BinanceFutureHttpClient(object):

    def __init__(self, api_key=None, secret=None, timeout=5):
        self.key = api_key
        self.secret = secret
        self.host = "https://fapi.binance.com"
        self.recv_window = 5000
        self.timeout = timeout
        self.order_count_lock = Lock()
        self.order_count = 1_000_000

    def build_parameters(self, params: dict):
        keys = list(params.keys())
        keys.sort()
        return '&'.join([f"{key}={params[key]}" for key in params.keys()])

    def request(self, req_method: RequestMethod, path: str, requery_dict=None, verify=False):
        url = self.host + path

        if verify:
            query_str = self._sign(requery_dict)
            url += '?' + query_str
        elif requery_dict:
            url += '?' + self.build_parameters(requery_dict)
        headers = {"X-MBX-APIKEY": self.key}

        response = requests.request(req_method.value, url=url, headers=headers, timeout=self.timeout)
        if response.status_code == 200:
            return response.status_code, response.json()
        else:
            try:
                return response.status_code, json.loads(response.text)
            except Exception as error:
                return response.status_code, {"msg": response.text, 'error': str(error)}

    def server_time(self):
        path = '/fapi/v1/time'
        return self.request(req_method=RequestMethod.GET, path=path)

    def exchangeInfo(self):
        path = '/fapi/v1/exchangeInfo'
        return self.request(req_method=RequestMethod.GET, path=path)

    def order_book(self, symbol, limit=5):
        limits = [5, 10, 20, 50, 100, 500, 1000]
        if limit not in limits:
            limit = 5

        path = "/fapi/v1/depth"
        query_dict = {"symbol": symbol,
                      "limit": limit
                      }

        return self.request(RequestMethod.GET, path, query_dict)

    def get_kline(self, symbol, interval: Interval, start_time=None, end_time=None, limit=500):
        """

        :param symbol:
        :param interval:
        :param start_time:
        :param end_time:
        :param limit:
        :return:
        [
            1499040000000,      // 开盘时间
            "0.01634790",       // 开盘价
            "0.80000000",       // 最高价
            "0.01575800",       // 最低价
            "0.01577100",       // 收盘价(当前K线未结束的即为最新价)
            "148976.11427815",  // 成交量
            1499644799999,      // 收盘时间
            "2434.19055334",    // 成交额
            308,                // 成交笔数
            "1756.87402397",    // 主动买入成交量
            "28.46694368",      // 主动买入成交额
            "17928899.62484339" // 请忽略该参数
        ]
        """
        path = "/fapi/v1/klines"

        query_dict = {
            "symbol": symbol,
            "interval": interval.value,
            "limit": limit
        }

        if start_time:
            query_dict['startTime'] = start_time

        if end_time:
            query_dict['endTime'] = end_time

        return self.request(RequestMethod.GET, path, query_dict)

    def get_latest_price(self, symbol):
        path = "/fapi/v1/ticker/price"
        query_dict = {"symbol": symbol}
        return self.request(RequestMethod.GET, path, query_dict)

    def get_ticker(self, symbol):
        path = "/fapi/v1/ticker/bookTicker"
        query_dict = {"symbol": symbol}
        return self.request(RequestMethod.GET, path, query_dict)

    ########################### the following request is for private data ########################

    def _timestamp(self):
        return int(time.time() * 1000)

    def _sign(self, params):

        requery_string = self.build_parameters(params)
        hexdigest = hmac.new(self.secret.encode('utf8'), requery_string.encode("utf-8"), hashlib.sha256).hexdigest()
        return requery_string + '&signature=' + str(hexdigest)

    def get_client_order_id(self):

        """
        generate the client_order_id for user.
        :return: new client order id
        """
        with self.order_count_lock:
            self.order_count += 1
            return "x-cLbi5uMH" + str(self._timestamp()) + str(self.order_count)

    def place_order(self, symbol: str, order_side: OrderSide, order_type: OrderType, quantity: Decimal, price: Decimal = None,
                time_inforce="GTC", client_order_id=None, recvWindow=5000, stop_price=0, position_side="BOTH", reduceOnly=False):
        path = '/fapi/v1/order'
        if client_order_id is None:
            client_order_id = self.get_client_order_id()

        params = {
            "symbol": symbol,
            "side": order_side.value,
            "type": order_type.value,
            "quantity": str(quantity),
            "recvWindow": recvWindow,
            "timestamp": self._timestamp(),
            "newClientOrderId": client_order_id,
            "positionSide": position_side,
            "reduceOnly": reduceOnly
        }

        if order_type == OrderType.LIMIT:
            if price is None:
                raise ValueError("price required for LIMIT")
            params['price'] = str(price)
            params['timeInForce'] = time_inforce
        elif order_type == OrderType.MARKET:
            if 'price' in params:
                del params['price']
        elif order_type == OrderType.STOP:
            if stop_price <= 0:
                raise ValueError("stopPrice must be greater than 0 for STOP")
            if price is None:
                raise ValueError("price required for STOP")
            params['price'] = str(price)
            params['stopPrice'] = str(stop_price)
        elif order_type == OrderType.MAKER:
            if price is None:
                raise ValueError("price required for MAKER")
            params['type'] = 'LIMIT'
            params['price'] = str(price)
            params['timeInForce'] = "GTX"

        return self.request(RequestMethod.POST, path=path, requery_dict=params, verify=True)

    def get_order(self, symbol, client_order_id=None):
        path = "/fapi/v1/order"
        query_dict = {"symbol": symbol, "timestamp": self._timestamp()}
        if client_order_id:
            query_dict["origClientOrderId"] = client_order_id

        return self.request(RequestMethod.GET, path, query_dict, verify=True)

    def cancel_order(self, symbol, client_order_id=None):
        path = "/fapi/v1/order"
        params = {"symbol": symbol, "timestamp": self._timestamp()}
        if client_order_id:
            params["origClientOrderId"] = client_order_id

        return self.request(RequestMethod.DELETE, path, params, verify=True)

    def get_open_orders(self, symbol=None):
        path = "/fapi/v1/openOrders"

        params = {"timestamp": self._timestamp()}
        if symbol:
            params["symbol"] = symbol

        return self.request(RequestMethod.GET, path, params, verify=True)

    def cancel_open_orders(self, symbol):
        """
        撤销某个交易对的所有挂单
        :param symbol: symbol
        :return: return a list of orders.
        """
        path = "/fapi/v1/allOpenOrders"

        params = {"timestamp": self._timestamp(),
                  "recvWindow": self.recv_window,
                  "symbol": symbol
                  }

        return self.request(RequestMethod.DELETE, path, params, verify=True)

    def get_balance(self):
        """
        [{'accountId': 18396, 'asset': 'USDT', 'balance': '530.21334791', 'withdrawAvailable': '530.21334791', 'updateTime': 1570330854015}]
        :return:
        """
        path = "/fapi/v1/balance"
        params = {"timestamp": self._timestamp()}

        return self.request(RequestMethod.GET, path=path, requery_dict=params, verify=True)

    def get_account_info(self):
        """
        {'feeTier': 2, 'canTrade': True, 'canDeposit': True, 'canWithdraw': True, 'updateTime': 0, 'totalInitialMargin': '0.00000000',
        'totalMaintMargin': '0.00000000', 'totalWalletBalance': '530.21334791', 'totalUnrealizedProfit': '0.00000000',
        'totalMarginBalance': '530.21334791', 'totalPositionInitialMargin': '0.00000000', 'totalOpenOrderInitialMargin': '0.00000000',
        'maxWithdrawAmount': '530.2133479100000', 'assets':
        [{'asset': 'USDT', 'walletBalance': '530.21334791', 'unrealizedProfit': '0.00000000', 'marginBalance': '530.21334791',
        'maintMargin': '0.00000000', 'initialMargin': '0.00000000', 'positionInitialMargin': '0.00000000', 'openOrderInitialMargin': '0.00000000',
        'maxWithdrawAmount': '530.2133479100000'}]}
        :return:
        """
        path = "/fapi/v1/account"
        params = {"timestamp": self._timestamp()}
        return self.request(RequestMethod.GET, path, params, verify=True)

    def get_position_info(self, symbol):
        """
        [{'symbol': 'BTCUSDT', 'positionAmt': '0.000', 'entryPrice': '0.00000', 'markPrice': '8326.40833498', 'unRealizedProfit': '0.00000000', 'liquidationPrice': '0'}]
        :return:

        if the symbol is not None, then return the following values:
        [{'symbol': 'ETHUSDT', 'positionAmt': '0.000', 'entryPrice': '0.0', 'markPrice': '3024.93000000',
        'unRealizedProfit': '0.00000000', 'liquidationPrice': '0', 'leverage': '25', 'maxNotionalValue': '1500000',
        'marginType': 'cross', 'isolatedMargin': '0.00000000', 'isAutoAddMargin': 'false', 'positionSide': 'BOTH',
        'notional': '0', 'isolatedWallet': '0', 'updateTime': 1649066944718}]
        """
        path = "/fapi/v2/positionRisk"
        params = {"timestamp": self._timestamp()}
        if symbol:
            params['symbol'] = symbol

        return self.request(RequestMethod.GET, path, params, verify=True)
    
    def set_leverage(self, symbol: str, leverage: int, recvWindow=5000):
        """
        Change user's initial leverage of a specific symbol.
        https://binance-docs.github.io/apidocs/futures/en/#change-initial-leverage-trade
        :param symbol: Trading symbol, e.g. BTCUSDT
        :param leverage: Target leverage; integer between 1-125
        :param recvWindow: The value must be less than 60000
        :return: (status_code, response_data)
        """
        if not 1 <= leverage <= 125:
            raise ValueError("leverage must be between 1 and 125")
        path = "/fapi/v1/leverage"
        params = {
            "symbol": symbol,
            "leverage": leverage,
            "recvWindow": recvWindow,
            "timestamp": self._timestamp()
        }
        return self.request(RequestMethod.POST, path=path, requery_dict=params, verify=True)

    def modify_order_price(self, symbol: str, client_order_id: str, new_price: Decimal, quantity: Decimal, recvWindow=5000):
        """
        Modify the price and quantity of an existing open LIMIT order.
        https://developers.binance.com/docs/derivatives/usds-margined-futures/trade/rest-api/Modify-Order
        :param symbol: Trading symbol, e.g. BTCUSDT
        :param client_order_id: Client order ID of the order to modify
        :param new_price: New limit price
        :param quantity: New quantity
        :param recvWindow: The value must be less than 60000
        :return: (status_code, response_data)
        """
        path = "/fapi/v1/order"
        params = {
            "symbol": symbol,
            "origClientOrderId": client_order_id,
            "price": str(new_price),
            "quantity": str(quantity),
            "recvWindow": recvWindow,
            "timestamp": self._timestamp()
        }
        return self.request(RequestMethod.PUT, path=path, requery_dict=params, verify=True)


    def close_all_positions(self, symbol: str, recvWindow=5000):
        """
        Close all positions for a given symbol by placing market orders in the opposite direction.
        Supports Hedge Mode (LONG and SHORT positions).
        :param symbol: Trading symbol, e.g. BTCUSDT
        :param recvWindow: The value must be less than 60000
        :return: List of (status_code, response_data) for each close order
        """
        status, positions = self.get_position_info(symbol=symbol)
        if status != 200 or not positions:
            return [(status, positions)]
        responses = []
        for position in positions:
            if position['symbol'] != symbol:
                continue
            position_amount = Decimal(position['positionAmt'])
            position_side = position.get('positionSide', 'BOTH')
            if position_amount > Decimal('0'):  # LONG
                status, order = self.place_order(
                    symbol=symbol,
                    order_side=OrderSide.SELL,
                    order_type=OrderType.MARKET,
                    quantity=position_amount,
                    price=None,
                    client_order_id=self.get_client_order_id(),
                    recvWindow=recvWindow,
                    position_side=position_side
                )
                responses.append((status, order))
            elif position_amount < Decimal('0'):  # SHORT
                status, order = self.place_order(
                    symbol=symbol,
                    order_side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=abs(position_amount),
                    price=None,
                    client_order_id=self.get_client_order_id(),
                    recvWindow=recvWindow,
                    position_side=position_side
                )
                responses.append((status, order))
        return responses if responses else [(200, {"msg": "No positions to close"})]

    def cancel_all_orders_for_symbol(self, symbol: str):
        """
        Cancel all open orders for a given symbol.
        https://developers.binance.com/docs/derivatives/usds-margined-futures/trade/rest-api/Cancel-All-Open-Orders
        :param symbol: Trading symbol, e.g. BTCUSDT
        :return: (status_code, response_data)
        """
        return self.cancel_open_orders(symbol=symbol)

    def close_all_and_cancel_all(self, symbol: str):
        """
        Close all positions and cancel all open orders for a given symbol.
        :param symbol: Trading symbol, e.g. BTCUSDT
        :return: Tuple of (close_positions_responses, cancel_orders_response)
        """
        close_responses = self.close_all_positions(symbol=symbol)
        cancel_response = self.cancel_all_orders_for_symbol(symbol=symbol)
        return close_responses, cancel_response

    def close_long_position(self, symbol: str, quantity: Decimal = None, percentage: Decimal = Decimal('1.0'), recvWindow=5000):
        """
        Close long position for a given symbol by placing a market SELL order.
        :param symbol: Trading symbol, e.g. BTCUSDT
        :param quantity: Quantity to close; if None, uses percentage
        :param percentage: Percentage of position to close (0.0 to 1.0)
        :param recvWindow: The value must be less than 60000
        :return: (status_code, response_data)
        """
        status, data = self.get_position_info(symbol=symbol)
        if status != 200 or not data:
            return status, data
        position = next((p for p in data if p['symbol'] == symbol and Decimal(p['positionAmt']) > 0), None)
        if not position:
            return 200, {"msg": "No long position to close"}
        position_amount = Decimal(position['positionAmt'])
        close_quantity = quantity if quantity is not None else position_amount * percentage
        if close_quantity > position_amount:
            close_quantity = position_amount
        status, order = self.place_order(
            symbol=symbol,
            order_side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=close_quantity,
            price=None,
            client_order_id=self.get_client_order_id(),
            recvWindow=recvWindow,
            position_side="LONG"
        )
        return status, order

    def close_short_position(self, symbol: str, quantity: Decimal = None, percentage: Decimal = Decimal('1.0'), recvWindow=5000):
        """
        Close short position for a given symbol by placing a market BUY order.
        :param symbol: Trading symbol, e.g. BTCUSDT
        :param quantity: Quantity to close; if None, uses percentage
        :param percentage: Percentage of position to close (0.0 to 1.0)
        :param recvWindow: The value must be less than 60000
        :return: (status_code, response_data)
        """
        status, data = self.get_position_info(symbol=symbol)
        if status != 200 or not data:
            return status, data
        position = next((p for p in data if p['symbol'] == symbol and Decimal(p['positionAmt']) < 0), None)
        if not position:
            return 200, {"msg": "No short position to close"}
        position_amount = Decimal(position['positionAmt'])
        close_quantity = quantity if quantity is not None else abs(position_amount) * percentage
        if close_quantity > abs(position_amount):
            close_quantity = abs(position_amount)
        status, order = self.place_order(
            symbol=symbol,
            order_side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=close_quantity,
            price=None,
            client_order_id=self.get_client_order_id(),
            recvWindow=recvWindow,
            position_side="SHORT"
        )
        return status, order

    def get_unrealized_pnl(self, symbol: str):
        """
        Get unrealized PNL for a given symbol.
        :param symbol: Trading symbol, e.g. BTCUSDT
        :return: Unrealized PNL as Decimal or None if position info cannot be retrieved.
        """
        position_info_status, position_info_data = self.get_position_info(symbol=symbol)
        if position_info_status == 200 and position_info_data and position_info_data[0]:
            return Decimal(position_info_data[0].get('unRealizedProfit', '0'))
        else:
            return None

    # Helper Functions
    def check_min_balance(self, min_balance: Decimal = Decimal('10')): # Example min balance of 10 USDT
        """
        Check if the available balance is above the minimum required balance.
        :param min_balance: Minimum balance required in USDT.
        :return: True if balance is above min_balance, False otherwise.
        """
        balance_status, balance_data = self.get_balance()
        if balance_status == 200 and balance_data:
            for asset_balance in balance_data:
                if asset_balance['asset'] == 'USDT':
                    available_balance = Decimal(asset_balance['withdrawAvailable'])
                    return available_balance >= min_balance
        return False

    def is_in_position(self, symbol: str):
        """
        Check if there is an open position for a given symbol.
        :param symbol: Trading symbol, e.g. BTCUSDT
        :return: True if in position, False otherwise.
        """
        position_info_status, position_info_data = self.get_position_info(symbol=symbol)
        if position_info_status == 200 and position_info_data and position_info_data[0]:
            position_amount = Decimal(position_info_data[0]['positionAmt'])
            return position_amount != Decimal('0')
        return False

    # Check can cascade - Functionality unclear, requires more context. Skipping for now.


    # Strategy Performance - Placeholder functions. Need actual historical trade data to calculate these.
    def calculate_sharpe_ratio(self, returns):
        """
        Calculate Sharpe Ratio. Placeholder, needs actual historical returns data.
        :param returns: List of returns (e.g., daily returns).
        :return: Sharpe Ratio.
        """
        if not returns or len(returns) < 2:
            return 0.0 # Not enough data

        import numpy as np
        returns_np = np.array(returns)
        excess_returns = returns_np - 0.0 # Assuming risk-free rate is 0 for simplicity
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0.0
        return sharpe_ratio

    def calculate_overall_pnl(self, trades):
        """
        Calculate overall PNL. Placeholder, needs actual historical trade data.
        :param trades: List of trade objects with PNL information.
        :return: Overall PNL.
        """
        if not trades:
            return Decimal('0')
        overall_pnl = sum([Decimal(trade['pnl']) for trade in trades], Decimal('0')) # Assuming trade object has 'pnl' field
        return overall_pnl

    def calculate_win_rate(self, trades):
        """
        Calculate win rate. Placeholder, needs actual historical trade data.
        :param trades: List of trade objects with PNL information.
        :return: Win rate (percentage).
        """
        if not trades:
            return 0.0
        winning_trades = sum(1 for trade in trades if Decimal(trade['pnl']) > 0) # Assuming trade object has 'pnl' field
        win_rate = (winning_trades / len(trades)) * 100 if trades else 0.0
        return win_rate

    def calculate_drawdown(self, equity_curve):
        """
        Calculate drawdown. Placeholder, needs equity curve data.
        :param equity_curve: List of equity values over time.
        :return: Maximum drawdown.
        """
        if not equity_curve or len(equity_curve) < 2:
            return Decimal('0')

        max_equity = equity_curve[0]
        max_drawdown = Decimal('0')
        for equity in equity_curve:
            max_equity = max(max_equity, equity)
            drawdown = (max_equity - equity) / max_equity if max_equity != 0 else 0 # Drawdown as percentage
            max_drawdown = max(max_drawdown, drawdown)
        return max_drawdown