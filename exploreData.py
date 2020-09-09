import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates
import datetime
from sklearn.model_selection import train_test_split#
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def convertTimestamp(df):
    df['Date'] = pd.to_datetime(df.Date)
    df['Year'] = df.Date.dt.year
    df['Month'] = df.Date.dt.month
    df['Day'] = df.Date.dt.day
    df['Hour'] = df.Date.dt.hour
    df['Minute'] = df.Date.dt.minute


def calcDelta(df):
    df = df.set_index('Date')
    df.sort_index(inplace=True)
    df['deltaDiesel'] = df.groupby(['Year', 'Month', 'Day'])[
        'Diesel'].transform(lambda x: (x - x[0]))
    df['deltaE5'] = df.groupby(['Year', 'Month', 'Day'])[
        'E5'].transform(lambda x: (x - x[0]))
    df['deltaE10'] = df.groupby(['Year', 'Month', 'Day'])[
        'E10'].transform(lambda x: (x - x[0]))
    df = df.reset_index(drop=False)
    return df


def addData(df1, df2):
    df1 = df1.set_index(['Year', 'Month', 'Day'])
    df2 = df2.set_index(['Year', 'Month', 'Day'])
    df1.sort_index(inplace=True)
    df2.sort_index(inplace=True)
    df1['Oil'] = df2.groupby('Date')['Price'].transform(lambda x: (x))

    return df1


dfOne = pd.DataFrame()
dfOne = pd.read_csv("prices_2020.csv", names=[
                    'Date', 'UUID', 'Diesel', 'E5', 'E10'])
dfTwo = pd.read_csv("brent_oil.csv", names=['Date', 'Price'])
convertTimestamp(dfOne)
convertTimestamp(dfTwo)
dfOne = calcDelta(dfOne)
# dfOne.reset_index(drop=False)
print(dfOne.dtypes)
dates = dates.date2num(dfOne.Date)
data = addData(dfOne, dfTwo)
data = data.reset_index(drop=False)
data = data.dropna(axis=0)
#data = data.drop(['Date'], axis=1)

labelenc = LabelEncoder()
onehotencoder = OneHotEncoder()
data['UUID'] = labelenc.fit_transform(data['UUID'])
#data = np.array(onehotencoder.fit_transform(data['UUID'])
print(data.head(25))
plt.scatter(data.UUID, data.deltaE10)
plt.show()
plt.scatter(data.Hour, data.deltaE10)
plt.show()
'''
y = data.deltaE10
X = data.drop(['deltaE10'], axis=1)
# X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.15, random_state = 42)
# dates = dates.date2num(X_test.Date)
plt.scatter(X_test.Date, y_test)
plt.show()
'''
