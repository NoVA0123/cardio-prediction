import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
import math
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)


df = pd.read_csv('../cardio-prediction/heart.csv')  # '..' means the directory in which you have cloned this repository
o2 = pd.read_csv('~/Downloads/o2Saturation.csv')
o2 = o2['98.6'][:303].tolist()
# o2_list = o2_list[:303]
df['o2'] = o2
print(df.head())

# finding gender ratio through pie chart
female = len(df[df['sex'] == 0])
male = len(df[df['sex'] == 1])
print(len(df), female, male)


# plotting the data
plt.pie([female, male], labels=['female', 'male'])
plt.show()


# Finding which anginal pain is more common(after excercise)
anginal_pain = df[df['exng'] == 1]
anginal_typical = len(anginal_pain[anginal_pain['cp'] == 1])
anginal_atypical = len(anginal_pain[anginal_pain['cp'] == 2])

# plotting the graph
plt.bar(['typical', 'atypical'], [anginal_typical, anginal_atypical])
plt.show()
print(f'Anginal typical = {anginal_typical}\n',
      f'Anginal atypical = {anginal_atypical}')

# analysing blood pressure
print(df['trtbps'].min(), df['trtbps'].max())


# Creating a function to create a series.
def heart_rate(data, column):

    minimum = data[column].min()
    maximum = df[column].max()

    remainder = (maximum-minimum) % 10
    next_ = minimum + remainder

    series = [minimum]
    f = True
    while f:
        if next_ == maximum:
            f = False
        series.append(next_)
        next_ += 10
    return series


heartrate = heart_rate(df, 'trtbps')
# print(heartrate)


# Creating a function to fit the data in accordance of series.
def heart_rate_calculator(data, column, series):

    dictionary = {}
    for x in range(1, len(series)):
        dictionary[f'{series[x - 1]}-{series[x]}'] = len(data[
            (data[column] >= series[x-1])
            &
            (data[column] <= series[x] - 1)])
    maximum = len(data[data[column] == data[column].max()])
    dictionary['190-200'] += maximum
    return dictionary


heart_rate_data = heart_rate_calculator(df, 'trtbps', heartrate)
# print(heart_rate_data)


# Plotting the data
names, values = [x for x in heart_rate_data.keys()],[x for x in heart_rate_data.values()]
for x in range(len(names)):
    print(f'{names[x]} = {values[x]}')
plt.bar(names, values)
plt.show()

# Applying ML model SVM

# Creating new variables for training and testing
X = df.drop(['output', 'oldpeak'], axis=1)
X = preprocessing.scale(X)
y = df['output']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Fitting the data
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)


# Finding accuracy
print(clf.score(X_test, y_test))
