import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates
import datetime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import joblib


def convertTimestamp(df):
    df['Date'] = pd.to_datetime(df.Date)
    df['Year'] = (df.Date.dt.year / 2020)
    df['Month'] = (df.Date.dt.month / 12)
    df['Day'] = (df.Date.dt.day / 30)
    df['Hour'] = (df.Date.dt.hour / 24)
    df['Minute'] = (df.Date.dt.minute / 60)


def calcDelta(df):
    df = df.set_index('Date')
    df.sort_index(inplace=True)
    df['deltaDiesel'] = df.groupby(['Year', 'Month', 'Day'])[
        'Diesel'].transform(lambda x: (x - x[0])*20)
    df['deltaE5'] = df.groupby(['Year', 'Month', 'Day'])[
        'E5'].transform(lambda x: (x - x[0])*20)
    df['deltaE10'] = df.groupby(['Year', 'Month', 'Day'])[
        'E10'].transform(lambda x: (x - x[0])*20)
    df = df.reset_index(drop=False)
    return df


def addData(df1, df2):
    df1 = df1.set_index(['Year', 'Month', 'Day'])
    df2 = df2.set_index(['Year', 'Month', 'Day'])
    df1.sort_index(inplace=True)
    df2.sort_index(inplace=True)
    df1['Oil'] = df2.groupby('Date')['Price'].transform(lambda x: (x/25))

    return df1


dfOne = pd.DataFrame()
dfOne = pd.read_csv("prices_2020.csv", names=[
                    'Date', 'UUID', 'Diesel', 'E5', 'E10'])
dfTwo = pd.read_csv("brent_oil.csv", names=['Date', 'Price'])
convertTimestamp(dfOne)
convertTimestamp(dfTwo)
dfOne = calcDelta(dfOne)

dates = dates.date2num(dfOne.Date)
data = addData(dfOne, dfTwo)
data = data.reset_index(drop=False)
data = data.dropna(axis=0)

uuid = pd.get_dummies(data.UUID)
data = pd.concat([data, uuid], axis=1)
data = data.drop(['Year', 'UUID', 'Diesel', 'E5', 'E10'], axis=1)


y = data.deltaE10
X = data.drop(['deltaDiesel', 'deltaE5', 'deltaE10'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


dat = X_test.Date
X_train = X_train.drop(['Date'], axis=1)
X_test = X_test.drop(['Date'], axis=1)
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

np.savetxt('X_train.csv', X_train, delimiter=',')
np.savetxt('X_test.csv', X_test, delimiter=',')
np.savetxt('y_train.csv', y_train, delimiter=',')
np.savetxt('y_test.csv', y_test, delimiter=',')
dat.to_csv('dates.csv')
'''

mlp = make_pipeline(StandardScaler(), MLPRegressor(
    max_iter=500, random_state=42))
hyperparameters = {'mlpregressor__hidden_layer_sizes': [
    (100, 10)], 'mlpregressor__solver': ['lbfgs'], 'mlpregressor__activation': ['relu']}
#
print("searching for parameters...")
clf = GridSearchCV(mlp, hyperparameters, cv=10)
print("training network...")
clf.fit(X_train, y_train)

# joblib.dump(clf, 'mlp_class.pkl')
print("predict values...")
y_pred = clf.predict(X_test)
y_pred = (y_pred/20)
y_test = (y_test/20)
print(mean_squared_error(y_test, y_pred))

plt.plot_date(dat, y_pred, linestyle='None', marker='.', color='r')
plt.plot_date(dat, y_test, linestyle='None', marker='x', color='b')
plt.show()
'''
