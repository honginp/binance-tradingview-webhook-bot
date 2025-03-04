# the following is the configuration for your tradingview webhook bot

# WEBHOOK_PASSPHRASE = "your password like"   # the password for security, must be the same from tradingview webhook settings.
# API_KEY = 'past your api secret here.'
# API_SECRET = 'past your api secret here.'

# the passphrase is a string for verifying that prevents others(especially the bad guy) to invoke the POST request in the program.
# passphrase 字符串是为了验证防止其他人(特别是坏人)去调用你的POST请求接口，这个接口是可以发送下单信号的。
import os

ENVIRONMENT = 'TEST' # 'REAL'
WEBHOOK_PASSPHRASE = "MyPassphrase"
API_KEY = os.environ.get('BINANCE_API_KEY1') if ENVIRONMENT == 'REAL' else os.environ.get('BINANCE_TEST_API_KEY2')
API_SECRET = os.environ.get('BINANCE_API_SECRET1') if ENVIRONMENT == 'REAL' else os.environ.get('BINANCE_TEST_API_SECRET2')
HOST = 'https://fapi.binance.com' if ENVIRONMENT == 'REAL' else 'https://testnet.binancefuture.com'

CANCEL_ORDERS_IN_SECONDS = 60 # every X second, will cancel your orders

QUERY_ORDERS_STATUS_IN_SECONDS = 5 # query orders' status for every five seconds.

# config your strategy name and the strategy data you want to trade here
#  tick_price: the price's precision, in Decimal
#  min_volume: the volume's precision, in Decimal
#  trading_volume: the amount of order you want to place here.
#  symbol: the binance symbol.
#  pos is the strategy's position in Decimal
# pls check out the price's precision and volume's precision from Binance Exchange.
from decimal import Decimal

# the amount should be wrapped with Decimal, or your order amount precision will be incorrect.
# 数量应该使用Decimal类来包裹起来，防止下单数量，价格精度的丢失。

strategies = {
    # strategy name -> strategy data
    "Binary": {
        'symbol': ['BTCUSDT'],
        'tick_price': Decimal("0.1"), # the price's precision, in Decimal
        'min_volume': Decimal("0.002"), # relates to min notional value which worth 200 USDT, so if price is $100k, the min volume should be 0.002
        'trading_volume': Decimal("0"),  # Set the amount of order you want to place here, use Decimal
        'pos': Decimal("0")  # current position when start your strategy, 策略当前的仓位, 用Decimal表示
    },
    "Coastline": {
        'symbol': ['BTCUSDT'],
        'tick_price': Decimal("0.1"), # the price's precision, in Decimal
        'min_volume': Decimal("0.002"), # relates to min notional value which worth 200 USDT, so if price is $100k, the min volume should be 0.002
        'trading_volume': Decimal("0"),  # 设置为你交易的数量，用Decimal表示. 
        'pos': Decimal("0")  # current position when start your strategy, 策略当前的仓位, 用Decimal表示
    }
}

# TRADING FEE
# 1. Commision Fee
# 0.0200% for maker (open/close by limit order)
# 0.0500% for taker (open/close by market order)
# 2. Funding Fee
# Positive funding fee means longs pay shorts by x% every 8 hours
# Negative funding fee means shorts pay longs by x% every 8 hours

