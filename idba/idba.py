import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

# --- Helper Functions ---

def calculate_osv(price, p_dcc, theta):
    """
    Calculate Overshoot Value (OSV), which measures how far the price has overshot a Directional Change point.
    :param price: Current market price
    :param p_dcc: Last Directional Change confirmation price
    :param theta: Threshold percentage for detecting a directional change
    :return: OSV value
    """
    return ((price - p_dcc) / p_dcc) / theta

def calculate_mdd(capital_series):
    """
    Calculate Maximum Drawdown (MDD), the maximum loss from a peak in capital.
    :param capital_series: Series of capital values over time
    :return: MDD as a negative percentage (e.g., -0.05 for 5% drawdown)
    """
    peak = capital_series.cummax()  # Cumulative maximum capital up to each point
    drawdown = (capital_series - peak) / peak  # Percentage drop from peak
    return drawdown.min()  # Minimum value (largest drawdown)

# --- IDBA Class ---

class IDBA:
    def __init__(self, data, theta_range, down_ind_range, capital=1000000):
        """
        Initialize the Intelligent Dynamic Backlash Agent (IDBA).
        :param data: DataFrame with 'time', 'bid', 'ask', 'mid' columns
        :param theta_range: List of theta values to optimize over (e.g., [0.001, 0.002])
        :param down_ind_range: List of down_ind values to optimize over (e.g., [-0.01, -0.02])
        :param capital: Initial capital (default: 1,000,000)
        """
        self.data = data
        self.theta_range = theta_range  # Range of DC thresholds to test
        self.down_ind_range = down_ind_range  # Range of overshoot indicators to test
        self.capital = capital  # Starting capital
        self.trades = []  # List to store trade history

    def train(self, train_data):
        """
        Training phase: Optimize theta and down_ind, train a C4.5 decision tree, and set MDD threshold.
        :param train_data: Subset of data for training
        :return: Best theta, best down_ind, trained classifier, PPV, FNR, and MDD from training
        """
        best_rr = -np.inf  # Best return rate (initialized to negative infinity)
        best_theta = None  # Best DC threshold
        best_down_ind = None  # Best overshoot indicator
        best_trades = None  # Best trade history

        # Step 1: Optimize theta and down_ind by simulating trades
        for theta in self.theta_range:
            for down_ind in self.down_ind_range:
                trades, final_capital = self.simulate_trading(
                    train_data, theta, down_ind, capital=self.capital, order_size=1.0
                )
                rr = (final_capital - self.capital) / self.capital * 100  # Return rate in percent
                if rr > best_rr:
                    best_rr = rr
                    best_theta = theta
                    best_down_ind = down_ind
                    best_trades = trades

        # Step 2: Train C4.5 decision tree for order size prediction
        X_train, y_train = self.prepare_training_data(best_trades)
        clf = DecisionTreeClassifier()  # Approximate C4.5 with scikit-learn
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_train)
        tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()  # Confusion matrix metrics
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive Predictive Value
        fnr = fn / (fn + tn) if (fn + tn) > 0 else 0  # False Negative Rate

        # Step 3: Calculate MDD from training trades
        capital_series = pd.Series([t['capital_after'] for t in best_trades])
        mdd_training = calculate_mdd(capital_series)

        return best_theta, best_down_ind, clf, ppv, fnr, mdd_training

    def prepare_training_data(self, trades):
        """
        Prepare training data for the C4.5 model using TBO and sigma_p as features.
        :param trades: List of trade dictionaries from simulation
        :return: Feature array (X) and target array (y)
        """
        X = []
        y = []
        for trade in trades:
            tbo = trade.get('tbo', 0)  # Time Between Overshoot and buy trigger (placeholder)
            sigma_p = trade.get('sigma_p', 0)  # Price volatility (placeholder)
            br = trade['profit'] > 0  # Binary return: True if profitable
            X.append([tbo, sigma_p])
            y.append(br)
        return np.array(X), np.array(y)

    def trade(self, trade_data, best_theta, best_down_ind, clf, ppv, fnr, mdd_training):
        """
        Trading phase: Execute trades with dynamic order sizing and MDD-based risk management.
        :param trade_data: Subset of data for trading
        :param best_theta: Optimized DC threshold
        :param best_down_ind: Optimized overshoot indicator
        :param clf: Trained C4.5 classifier
        :param ppv: Positive Predictive Value from training
        :param fnr: False Negative Rate from training
        :param mdd_training: MDD threshold from training
        :return: Trade history and final capital
        """
        capital = self.capital  # Current capital
        position = 0  # Number of units held (in price terms)
        entry_price = None  # Price at which position was opened
        trades = []  # Trade history
        capital_series = [capital]  # Track capital over time

        # Simplified DC variables (in practice, these would be dynamically updated)
        p_dcc = trade_data['mid'].iloc[0]  # Initial Directional Change confirmation price
        p_dcc_up = p_dcc * (1 + best_theta)  # Upper DC threshold for sell signal

        for i in range(len(trade_data)):
            row = trade_data.iloc[i]
            price = row['mid']  # Current mid price
            time = row['time']

            if position == 0:  # No open position, look for buy signal
                osv = calculate_osv(price, p_dcc, best_theta)
                if osv <= best_down_ind:  # Buy condition (Rule DBA.1)
                    # Placeholder for TBO and sigma_p (replace with real calculations)
                    tbo = 10  # Example: time between extreme point and buy trigger
                    sigma_p = 0.005  # Example: price volatility
                    fbr = clf.predict([[tbo, sigma_p]])[0]  # Forecasted Binary Return
                    order_size = ppv if fbr else fnr  # Dynamic order size
                    buy_size = order_size * capital  # Capital to invest
                    position = buy_size / price  # Units bought
                    entry_price = price
                    trades.append({
                        'time': time, 'action': 'buy', 'price': price, 'capital_before': capital
                    })
                    capital -= buy_size

            elif position > 0:  # Open position, look for sell signal
                if price >= p_dcc_up:  # Sell condition (Rule DBA.2)
                    sell_price = price
                    profit = position * (sell_price - entry_price)  # Profit from trade
                    capital += position * sell_price  # Update capital
                    trades.append({
                        'time': time, 'action': 'sell', 'price': sell_price,
                        'profit': profit, 'capital_after': capital
                    })
                    position = 0
                    capital_series.append(capital)

                    # Check MDD for risk management
                    mdd_trading = calculate_mdd(pd.Series(capital_series))
                    if mdd_trading < mdd_training:  # MDD exceeds threshold (more negative)
                        print("MDD exceeded, halting trading for retraining...")
                        break

        return trades, capital

    def simulate_trading(self, data, theta, down_ind, capital, order_size):
        """
        Simulate trading to optimize parameters during training.
        :param data: Data subset for simulation
        :param theta: DC threshold to test
        :param down_ind: Overshoot indicator to test
        :param capital: Starting capital
        :param order_size: Fixed order size for simulation
        :return: Trade history and final capital
        """
        trades = []
        position = 0
        entry_price = None
        p_dcc = data['mid'].iloc[0]  # Initial DC confirmation price
        p_dcc_up = p_dcc * (1 + theta)  # Upper DC threshold

        for i in range(len(data)):
            price = data['mid'].iloc[i]
            time = data['time'].iloc[i]

            if position == 0:
                osv = calculate_osv(price, p_dcc, theta)
                if osv <= down_ind:
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

        # Ensure all trades have required fields for training
        for t in trades:
            if 'capital_after' not in t:
                t['capital_after'] = capital
            if 'profit' not in t:
                t['profit'] = 0
            t['tbo'] = 10  # Placeholder
            t['sigma_p'] = 0.005  # Placeholder

        return trades, capital

# --- Usage Example ---

# Load sample data (replace with your own forex data)
data = pd.read_csv('forex_data.csv')  # Expected columns: 'time', 'bid', 'ask', 'mid'

# Define parameter ranges
theta_range = [0.001, 0.002, 0.005, 0.01]  # DC thresholds to test
down_ind_range = np.arange(-0.01, -0.05, -0.01)  # Overshoot indicators to test

# Split data into training and trading periods
train_data = data.iloc[:len(data)//2]
trade_data = data.iloc[len(data)//2:]

# Initialize IDBA
idba = IDBA(data, theta_range, down_ind_range, capital=1000000)

# Train the model
best_theta, best_down_ind, clf, ppv, fnr, mdd_training = idba.train(train_data)

# Execute trades
trades, final_capital = idba.trade(trade_data, best_theta, best_down_ind, clf, ppv, fnr, mdd_training)

# Output results
print(f"Best Theta: {best_theta}, Best Down Ind: {best_down_ind}")
print(f"Final Capital: {final_capital}")
print(f"Number of Trades: {len(trades)}")