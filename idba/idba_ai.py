import pandas as pd
import numpy as np
from keras.models import Sequential
from keras import layers
from sklearn.metrics import confusion_matrix

class IDBA:
    def __init__(self, data, theta_range, down_ind_range, capital=1000000):
        self.data = data
        self.theta_range = theta_range
        self.down_ind_range = down_ind_range
        self.capital = capital
        self.trades = []

    def calculate_osv(self, price, p_dcc, theta):
        return ((price - p_dcc) / p_dcc) / theta

    def calculate_mdd(self, capital_series):
        peak = capital_series.cummax()
        drawdown = (capital_series - peak) / peak
        return drawdown.min()

    def simulate_trading(self, data, theta, down_ind, capital, order_size):
        trades = []
        position = 0
        entry_price = None
        p_dcc = data['mid'].iloc[0]
        p_dcc_up = p_dcc * (1 + theta)

        for i in range(len(data)):
            price = data['mid'].iloc[i]
            time = data['time'].iloc[i]

            if position == 0:
                osv = self.calculate_osv(price, p_dcc, theta)
                if osv <= down_ind:
                    buy_size = order_size * capital
                    position = buy_size / price
                    entry_price = price
                    trades.append({
                        'time': time, 'action': 'buy', 'price': price, 'capital_before': capital,
                        'tbo': 10, 'sigma_p': 0.005, 'mean_price': np.mean(data['mid'].iloc[:i]),
                        'min_price': np.min(data['mid'].iloc[:i]), 'max_price': np.max(data['mid'].iloc[:i]),
                        'num_dc_events': 5  # Placeholder
                    })
                    capital -= buy_size

            elif position > 0:
                if price >= p_dcc_up:
                    profit = position * (price - entry_price)
                    capital += position * price
                    trades[-1]['capital_after'] = capital
                    trades[-1]['profit'] = profit
                    position = 0

        return trades, capital

    def train(self, train_data):
        best_rr = -np.inf
        best_theta = None
        best_down_ind = None
        best_trades = None

        for theta in self.theta_range:
            for down_ind in self.down_ind_range:
                trades, final_capital = self.simulate_trading(train_data, theta, down_ind, self.capital, 1.0)
                rr = (final_capital - self.capital) / self.capital * 100
                if rr > best_rr:
                    best_rr = rr
                    best_theta = theta
                    best_down_ind = down_ind
                    best_trades = trades

        X_train, y_train = self.prepare_training_data(best_trades)
        model = Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        y_pred = model.predict(X_train) > 0.5
        tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        fnr = fn / (fn + tn) if (fn + tn) > 0 else 0

        capital_series = pd.Series([t['capital_after'] for t in best_trades])
        mdd_training = self.calculate_mdd(capital_series)

        return best_theta, best_down_ind, model, ppv, fnr, mdd_training

    def prepare_training_data(self, trades):
        X = []
        y = []
        for trade in trades:
            tbo = trade.get('tbo', 0)
            sigma_p = trade.get('sigma_p', 0)
            mean_price = trade.get('mean_price', 0)
            min_price = trade.get('min_price', 0)
            max_price = trade.get('max_price', 0)
            num_dc_events = trade.get('num_dc_events', 0)
            br = trade['profit'] > 0 if 'profit' in trade else False
            features = [tbo, sigma_p, mean_price, min_price, max_price, num_dc_events]
            X.append(features)
            y.append(br)
        return np.array(X), np.array(y)

    def trade(self, trade_data, best_theta, best_down_ind, model, ppv, fnr, mdd_training):
        capital = self.capital
        position = 0
        entry_price = None
        p_dcc = trade_data['mid'].iloc[0]
        p_dcc_up = p_dcc * (1 + best_theta)
        trades = []
        capital_series = [capital]

        for i in range(len(trade_data)):
            row = trade_data.iloc[i]
            price = row['mid']
            time = row['time']

            if position == 0:
                osv = self.calculate_osv(price, p_dcc, best_theta)
                if osv <= best_down_ind:
                    tbo = 10  # Placeholder
                    sigma_p = 0.005  # Placeholder
                    mean_price = np.mean(trade_data['mid'].iloc[:i])
                    min_price = np.min(trade_data['mid'].iloc[:i])
                    max_price = np.max(trade_data['mid'].iloc[:i])
                    num_dc_events = 5  # Placeholder
                    features = [tbo, sigma_p, mean_price, min_price, max_price, num_dc_events]
                    fbr = model.predict(np.array([features]))[0][0] > 0.5
                    order_size = ppv if fbr else fnr
                    buy_size = order_size * capital
                    position = buy_size / price
                    entry_price = price
                    trades.append({
                        'time': time, 'action': 'buy', 'price': price, 'capital_before': capital
                    })
                    capital -= buy_size

            elif position > 0:
                if price >= p_dcc_up:
                    profit = position * (price - entry_price)
                    capital += position * price
                    trades.append({
                        'time': time, 'action': 'sell', 'price': price,
                        'profit': profit, 'capital_after': capital
                    })
                    position = 0
                    capital_series.append(capital)

                    mdd_trading = self.calculate_mdd(pd.Series(capital_series))
                    if mdd_trading < mdd_training:
                        break

        return trades, capital

# Usage example
data = pd.read_csv('forex_data.csv')  # Expected columns: 'time', 'bid', 'ask', 'mid'
theta_range = [0.001, 0.002, 0.005, 0.01]
down_ind_range = np.arange(-0.01, -0.05, -0.01)
train_data = data.iloc[:len(data)//2]
trade_data = data.iloc[len(data)//2:]
idba = IDBA(data, theta_range, down_ind_range, capital=1000000)
best_theta, best_down_ind, model, ppv, fnr, mdd_training = idba.train(train_data)
trades, final_capital = idba.trade(trade_data, best_theta, best_down_ind, model, ppv, fnr, mdd_training)
print(f"Final Capital: {final_capital}")