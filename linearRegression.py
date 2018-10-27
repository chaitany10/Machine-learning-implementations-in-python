import math
import quandl
from sklearn import preprocessing, svm, model_selection
from sklearn.linear_model import LinearRegression

'''
made a dataframe by taking stock info from quandl
only included open,close,low,high,volume for analysis
now defining addditional rows from existing to have more insights into data
'''
df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. Close', 'Adj. Low', 'Adj. High',
         'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

'''making a forecast column ..currently taking adj. close as a label.
Then for outlier detection making all non available values as some large negative value to detect it
then selecting some variable forecast_out which will shift the adj.close up by some factor currently using 0.01*len(df)
making a new row as label with shifted values
'''

forecast_col = 'Adj. Close'
df.fillna('-99999', inplace=True)

forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

#print(df.head())
'''
We drop label row from X as it is part of y and add label to y.Then we scale X. Then w split the data into 
training and testing data

'''

X = np.array(df.drop(['label'],1))
y = np.array(df['label'])

X = preprocessing.scale(X)

df.dropna(inplace=True)
y= np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)﻿

