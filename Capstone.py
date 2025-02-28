import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

pd.set_option('display.width', 0)
pd.set_option('display.max_columns', None)

# Use pandas to import data as a DataFrame
data = pd.read_csv('spy.csv', index_col=0, parse_dates = True)
#print(data)

#Convert index of data to datetime
data.index = pd.to_datetime(data.index)

#Create a filter for 2006 data and onwards, since the 2X leveraged ETF did not exist until 2006
leveraged_data = data.loc[data.index >= '2006-01-01'].copy()
print(leveraged_data)

# Calculate the 200-day simple moving average, which is simply the average of the closing price for 200 days.
data['SMA200'] = data['Close'].rolling(window=200).mean()
print(data)

#Since the 200-day SMA is subject to a lot of false positives in sideways markets, change the signal to only activate when
#above or below 200-day SMA for over a week
data['Above_200_SMA'] = data['Close'] > data['SMA200']
data['Above_200_SMA_7D'] = data['Above_200_SMA'].rolling(window=7).sum() == 7  # True if all 7 days were above
data['Below_200_SMA_7D'] = data['Above_200_SMA'].rolling(window=7).sum() == 0   # True if all 7 days were below


# Plot the data using matplotlib to visualize how the price has moved over time
features = ['Open','High','Low','Close','Volume', 'SMA200']
data[features].plot(subplots=True,figsize=(12,10),title='S&P500 SPY',linestyle='-',linewidth=2)
plt.show()

# Generate signals based on crossovers
# create a signal column: 1 if price > SMA200, -1 if price < SMA200.
#Initialize signal to 1 since the 200D SMA does not exist in the first 200 days of data
data['Signal'] = 1
data = data
data.loc[(data['Close'] > data['SMA200']) & (data['Above_200_SMA_7D']), 'Signal'] = 1
data.loc[(data['Close'] < data['SMA200']) & (data['Below_200_SMA_7D']), 'Signal'] = -1

#To capture only the crossovers compare today's signal to yesterday's
data['Prev_Signal'] = data['Signal'].shift(1)
# Create a Trade column: 1 for a buy signal (crossing from below to above),
# -1 for a sell signal (crossing from above to below)
data['Trade'] = 0
data.loc[(data['Signal'] == 1) & (data['Prev_Signal'] == -1), 'Trade'] = 1  # Buy signal
data.loc[(data['Signal'] == -1) & (data['Prev_Signal'] == 1), 'Trade'] = -1  # Sell signal

# 4. Define outcomes for each trade.
#    For example, measure the return over the next N days (e.g., 10 days).
N = 100
data['Future_Return'] = data['Close'].shift(-N) / data['Close'] - 1

#    Label the trade outcome:
#    For a buy, a positive return might be considered a success (1).
#    For a sell, a negative return might be considered a success (1).
def label_outcome(row):
    if row['Trade'] == 1:
        return 1 if row['Future_Return'] > 0 else 0
    elif row['Trade'] == -1:
        return 1 if row['Future_Return'] < 0 else 0
    else:
        return np.nan

data['Outcome'] = data.apply(label_outcome, axis=1)
# Remove rows without a trade signal or outcome
data.dropna(subset=['Outcome'], inplace=True)

# Machine Learning Portion -------------------------------------------------------------
#    Create a feature like the difference between the current price and the SMA.
data['Price_SMA_Diff'] = data['Close'] - data['SMA200']
#    You can add more features as needed.

features = ['Close', 'SMA200', 'Price_SMA_Diff']
X = data[features]
y = data['Outcome'].astype(int)

# split the data into training and test sets
# Here we do a time-series split by not shuffling the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# train a Random Forest Classifier
#n_estimators = number of decision trees, random_state = seed for randomness model
clf = RandomForestClassifier(n_estimators=500, random_state=42)
clf.fit(X_train, y_train)

# 8. Make predictions and evaluate the model
y_pred = clf.predict(X_test)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#Plot feature importances
importances = clf.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(8, 6))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='blue', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# Daily spy returns
data['Daily_Returns'] = data['Close'].pct_change()
# Cumulative spy return
data['Buy_Hold'] = (1 + data['Daily_Returns']).cumprod()
#Simulate portfolio value starting with $10,000
data['Buy_Hold_Portfolio'] = 10000*data['Buy_Hold']


# Find the returns if we follow the 200-D SMA strategy -----------------------------------------
data['Position'] = data['Signal'].shift(1).fillna(0).apply(lambda x: 1 if x == 1 else 0)
# When position = 0, daily return is 0

# Calculate the 200D SMA strategies returns
data['Strategy_Return'] = data['Daily_Returns'] * data['Position']
# Calculate cumulative return for the strategy
data['Strategy_Cumulative_Return'] = (1 + data['Strategy_Return']).cumprod()
#Simulate portfolio value for the 200D SMA strategy starting with $10,000
data['Strategy_Portfolio'] = 10000 * data['Strategy_Cumulative_Return']
# --------------------------------------------------------------------------------------------

#Leveraged returns from 2006 onward
# Daily 2Xspy returns
data['2X_Returns'] = data['Close'].pct_change()*2
# Cumulative 2Xspy return
#data['Buy_Hold'] = (1 + data['2X_Returns']).cumprod()


# Simulate a 2X leveraged S&P ETF, such as SSO when used with the 200-Day SMA strategy
data['2X_Strategy_Return'] = data['2X_Returns'] * data['Position']
data['2X_Strategy_Cumulative_Return'] = (1 + data['2X_Strategy_Return']).cumprod()
#Simulate portfolio value for the 200D SMA strategy starting with $10,000
data['2X_Strategy_Portfolio'] = 10000 * data['2X_Strategy_Cumulative_Return']


# Simulate a 2X leveraged S&P ETF, such as SSO when bought and held (note: SSO did not exist until 2006)
data['2X_Daily_Returns'] = data['2X_Returns']
data['2X_Buy_Hold'] = (1 + data['2X_Daily_Returns']).cumprod()
data['2X_Buy_Hold_Portfolio'] = 10000 * data['2X_Buy_Hold']



# Plot for comparison
plt.figure(figsize=(12,6))
plt.plot(data.index, data['Buy_Hold'], label='Buy & Hold')
plt.plot(data.index, data['Strategy_Cumulative_Return'], label='200-Day SMA Based Trading Strategy')
plt.title('Comparison of Buy & Hold vs 200-Day SMA Based Trading Strategy Cumulative Returns')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()

# Plot the simulated portfolios of all 3 starting with $10,000
plt.figure(figsize=(12,6))
plt.plot(data.index, data['Buy_Hold_Portfolio'], label='Buy & Hold Portfolio Value')
plt.plot(data.index, data['Strategy_Portfolio'], label='200-Day SMA Strategy Portfolio Value')
plt.plot(data.index, data['2X_Buy_Hold_Portfolio'], label='2X Leveraged Buy and Hold Portfolio Value')
plt.plot(data.index, data['2X_Strategy_Portfolio'], label = '2X Leveraged 200-Day SMA Strategy Portfolio Value')
plt.title('Portfolio Value Comparison: $10,000 Investment')
plt.xlabel('Date')
plt.ylabel('Portfolio Value ($)')
plt.legend()
plt.show()


# Calculate and plot max drawdown for both strategies ----------------------------------------------

# For Buy & Hold: Calculate the running maximum and drawdown
data['Buy_Hold_Running_Max'] = data['Buy_Hold'].cummax()
data['Buy_Hold_Drawdown'] = data['Buy_Hold'] / data['Buy_Hold_Running_Max'] - 1

# For the 200-Day SMA Strategy: Calculate the running maximum and drawdown
data['Strategy_Running_Max'] = data['Strategy_Cumulative_Return'].cummax()
data['Strategy_Drawdown'] = data['Strategy_Cumulative_Return'] / data['Strategy_Running_Max'] - 1

# --------------------------------------------------------------------------------------------------------


# Plot the drawdown curves
plt.figure(figsize=(12,6))
plt.plot(data.index, data['Buy_Hold_Drawdown'], label='Buy & Hold Drawdown', color='blue')
plt.plot(data.index, data['Strategy_Drawdown'], label='200-Day SMA Strategy Drawdown', color='red')
plt.title('Drawdown Comparison')
plt.xlabel('Date')
plt.ylabel('Drawdown')
plt.legend()
plt.show()

# User recommendations
# Produce a buy/hold recommendation if SPY is above the 200-day SMA,
# and a sell recommendation if it closed below the 200-day SMA.
data['Recommendation'] = np.where(data['Close'] > data['SMA200'], 'Buy/Hold', 'Sell')

# Display the latest recommendations
print(data[['Close', 'SMA200', 'Recommendation']])




'''
example of matplotlib
x = np.linspace(0, 2 * np.pi, 200)
y = np.sin(3*x)
y = np.sin(3*x)

fig, ax = plt.subplots()
ax.plot(x, y)
plt.show()
'''

#TODO
#Fix UI so you don't have to scroll to see draw down chart ------- Done

#Add 2X leveraged charts just for fun
#Add more to the ML part of the project