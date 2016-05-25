## MTH 9899 HW3 Victor Istratov

import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

runProblem1 = True
runProblem2 = True
runProblem3 = True

inputfile = "train.csv"

dataset = pd.read_csv(inputfile)

for name in ['Ret_MinusTwo', 'Ret_MinusOne']+['Ret_{}'.format(i) for i in range(2,181)]:
    dataset[name] = dataset[name] * 10000

dataset = dataset.fillna(0)

dataset['Ret_MinusZero'] = 0
for i in range(121,181):
    dataset['Ret_MinusZero'] =  dataset['Ret_MinusZero'].add(dataset['Ret_{}'.format(i)], fill_value=0)


#X = dataset[['Feature_5','Feature_7','Ret_MinusTwo', 'Ret_MinusOne']+['Ret_{}'.format(i) for i in range(2,121)]]
#X = dataset[['Feature_5','Feature_7']]
X = dataset[['Feature_5','Feature_7','Ret_MinusTwo', 'Ret_MinusOne']]

## tried using different independent variables, but it only gets worse

from sklearn.datasets import load_boston
from sklearn.cross_validation import cross_val_score

boston = load_boston()
regressor = DecisionTreeRegressor(random_state=0)

regressor.fit(boston.data, boston.target)
score1 = regressor.score(boston.data, boston.target)

testr = cross_val_score(regressor, boston.data, boston.target, cv=10)

print(testr)




Y = dataset['Ret_MinusZero']


## test fit
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
regr_1 = DecisionTreeRegressor(max_depth=30)
regr_1.fit(X_train, Y_train)
score1 = regr_1.score(X_train, Y_train)
score2 = regr_1.score(X_test, Y_test)
print('score_train : {}, score_test : {}'.format( score1, score2))
#
# regr_1.fit(X, Y)
# score1 = regr_1.score(X, Y)
# print('score_test : {}'.format( score1))




if(runProblem1):
    #Problem 1
    #part 1

    regr_1 = DecisionTreeRegressor()
    regr_1.fit(X, Y)

    score = regr_1.score(X, Y)

    print('Problem 1 part 1 score : {}'.format(score))

    #part 2

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    regr_1 = DecisionTreeRegressor(max_depth=4)

    regr_1.fit(X_train,Y_train)


    score = regr_1.score(X_train, Y_train)
    print('Problem 1 part 2 Train score : {}'.format(score))
    score = regr_1.score(X_test, Y_test)
    print('Problem 1 part 2 Test score : {}'.format(score))


    #part 3

    tr_depth = []
    tr_score = []
    num_results = 0

    for max_features in [2]: #[1,2]
        for max_depth in range(2,50): #
            for min_samples_leaf in [1,2,3]: # [1,2,3,4,5]
                regr_1 = DecisionTreeRegressor(max_features=max_features, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
                regr_1.fit(X_train, Y_train)
                score1 = regr_1.score(X_train, Y_train)
                score2 = regr_1.score(X_test, Y_test)
                print('Features: {}, max_depth: {}, min_samples_leaf: {}, score_train : {}, score_test : {}'.format(max_features, max_depth, min_samples_leaf, score1, score2))
                if(max_features==2 and min_samples_leaf==1):
                    tr_depth.append(max_depth)
                    tr_score.append([score1,score2])

    tr_score = np.array(tr_score)
    # Plot the results
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(tr_depth, tr_score[:,0], c="b", label="train")
    ax1.scatter(tr_depth, tr_score[:,1], c="g", label="test")


    # pick the best

    regr_1 = DecisionTreeRegressor(max_depth=5, min_samples_leaf=1)
    regr_1.fit(X_train, Y_train)
    Y_predict = regr_1.predict(X_test)
    export_graphviz(regr_1)

if(runProblem2):
    # Problem 2

    Y1 = []

    for i in range(50):
        regr_1 = DecisionTreeRegressor(max_depth=20, min_samples_leaf=2)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        regr_1.fit(X_train, Y_train)
        Y1.append(regr_1.predict(X_test))

    # part 1
    Y_mean = np.mean(Y1, axis=0)

    # part 2
    Y_median = np.median(Y1, axis=0)

    # part 3
    rf = RandomForestRegressor(max_depth=20, min_samples_leaf=2)
    rf.fit(X_train, Y_train)
    Y_rf = rf.predict(X_test)

    score = rf.score(X_test, Y_test)
    print('Problem 2 part 3 Train score : {}'.format(score))


    # part 4

    gbr = GradientBoostingRegressor(max_depth=20, min_samples_leaf=3)

    gbr.fit(X_train, Y_train)
    Y_gbr = gbr.predict(X_test)

    score = gbr.score(X_test, Y_test)
    print('Problem 2 part 4 Train score : {}'.format(score))



if(runProblem3):
    from keras.models import Sequential
    from keras.layers.core import Activation, Dense
    from keras.callbacks import EarlyStopping


    X = dataset[['Feature_5', 'Feature_7','Ret_MinusTwo', 'Ret_MinusOne']+['Ret_{}'.format(i) for i in range(2,121)]]
    Y = dataset['Ret_MinusZero']

    X['Feature_5'] = (X['Feature_5'] - np.mean(X['Feature_5']))/np.std(X['Feature_5'])
    X['Feature_7'] = (X['Feature_7'] - np.mean(X['Feature_7']))/np.std(X['Feature_7'])

    # Part 1
    model = Sequential()
    model.add(Dense(123, input_dim=123))
    model.add(Dense(123))
    model.add(Dense(123))
    model.add(Activation('relu'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    earlystopper = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
    history = model.fit(X.values, Y.values, batch_size=100, nb_epoch=10, validation_split=.15) #, callbacks=[earlystopper])
    print("Done training " + str(datetime.datetime.now()))

    # Part 2
    model = Sequential()
    model.add(Dense(123, input_dim=123))
    model.add(Dense(250))
    model.add(Dense(250))
    model.add(Activation('relu'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    earlystopper = EarlyStopping(monitor='val_loss', patience=1, verbose=1)
    history = model.fit(X.values, Y.values, batch_size=100, nb_epoch=10, validation_split=.15)
    print("Done training " + str(datetime.datetime.now()))