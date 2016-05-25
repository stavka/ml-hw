## MTH 9899 HW3 Victor Istratov

import datetime
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

runProblem1 = False
runProblem2 = True
runProblem3 = False

#inputfile = "5192.csv"

path = r'data/'
all_files = glob.glob(os.path.join(path, "*.csv"))
dataset = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

#dataset = pd.read_csv(inputfile)

#for name in ['Ret_MinusTwo', 'Ret_MinusOne']+['Ret_{}'.format(i) for i in range(2,181)]:
#    dataset[name] = dataset[name] * 10000



#dataset = dataset.fillna(0)

#dataset['Ret_MinusZero'] = 0
#for i in range(121,181):
#    dataset['Ret_MinusZero'] =  dataset['Ret_MinusZero'].add(dataset['Ret_{}'.format(i)], fill_value=0)


#X = dataset[['Feature_5','Feature_7','Ret_MinusTwo', 'Ret_MinusOne']+['Ret_{}'.format(i) for i in range(2,121)]]
#X = dataset[['Feature_5','Feature_7']]

X = dataset[['OrderQty','f1','f2','f3','time','Slippage',
             #'side',
             'volEstimate',
             #'expectedVolume',
             # 'volume5',
             'volume4','volume3','volume2','volume1',
              #'spread5',
              'spread4','spread3','spread2','spread1',
              #'avgAskQty5',
              'avgAskQty4','avgAskQty3','avgAskQty2','avgAskQty1',
              #'avgBidQty5',
               'avgBidQty4','avgBidQty3','avgBidQty2','avgBidQty1',
               #'abnormalVolume5',
               'abnormalVolume4','abnormalVolume3','abnormalVolume2','abnormalVolume1',
              #'residVol5',
               'residVol4','residVol3','residVol2','residVol1']]

X['OrderQty'] = X['OrderQty'] / dataset['expectedVolume']



for colname in X.columns.values:
   X[colname] = (X[colname] - np.mean(X[colname])) / np.std(X[colname])



X = pd.concat([X,pd.get_dummies(dataset['side'])], axis=1)

X = X.dropna(how='any')

Y = X['Slippage']

X.drop('Slippage', axis=1, inplace=True)



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

    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

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

    #for max_features in [2]: #[1,2]
    for max_depth in range(2,25): #
        for min_samples_leaf in [1,2,3]: # [1,2,3,4,5]
            regr_1 = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
            regr_1.fit(X_train, Y_train)
            score1 = regr_1.score(X_train, Y_train)
            score2 = regr_1.score(X_test, Y_test)
            print('max_depth: {}, min_samples_leaf: {}, score_train : {}, score_test : {}'.format(max_depth, min_samples_leaf, score1, score2))
            if(min_samples_leaf==1):
                tr_depth.append(max_depth)
                tr_score.append([score1,score2])

    tr_score = np.array(tr_score)
    # Plot the results
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.scatter(tr_depth, tr_score[:,0], c="b", label="train")
    ax1.scatter(tr_depth, tr_score[:,1], c="g", label="test")


    # pick the best

    regr_1 = DecisionTreeRegressor(max_depth=8, min_samples_leaf=2)
    regr_1.fit(X_train, Y_train)
    Y_predict = regr_1.predict(X_test)
    export_graphviz(regr_1)

if(runProblem2):
    # Problem 2

#    Y1 = []
#
#    for i in range(50):
#        regr_1 = DecisionTreeRegressor(max_depth=8, min_samples_leaf=2)
#        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
#        regr_1.fit(X_train, Y_train)
#        Y1.append(regr_1.predict(X_test))
#
#    # part 1
#    Y_mean = np.mean(Y1, axis=0)
#
#    # part 2
#    Y_median = np.median(Y1, axis=0)

    # part 3
    rf = RandomForestRegressor(max_depth=8, min_samples_leaf=2)
    rf.fit(X_train, Y_train)
    Y_rf = rf.predict(X_test)

    score = rf.score(X_test, Y_test)
    print('Problem 2 part 3 Test score : {}'.format(score))


    # part 4

    gbr = GradientBoostingRegressor(max_depth=8, min_samples_leaf=2 ) 
                                  #  loss='huber', learning_rate=0.1,
                                  #  n_estimators=100 )

    gbr.fit(X_train, Y_train)
    Y_gbr = gbr.predict(X_test)

    score = gbr.score(X_test, Y_test)
    print('Problem 2 part 4 Test score : {}'.format(score))
    
    
    
    etr = ExtraTreesRegressor(n_estimators=100, max_depth=8,min_samples_leaf=2 )
    
    etr.fit(X_train, Y_train)
    
    Y_etr = etr.predict(X_test)    
    score = r2_score(Y_test.values, Y_etr)
    print('Problem 2 part 5a Test score : {}'.format(score))
    

    score = etr.score(X_test, Y_test)
    print('Problem 2 part 5b Test score : {}'.format(score))
    



if(runProblem3):
    from keras.models import Sequential
    from keras.layers.core import Activation, Dense, Dropout
    from keras.callbacks import EarlyStopping


    #X = dataset[['Feature_5', 'Feature_7','Ret_MinusTwo', 'Ret_MinusOne']+['Ret_{}'.format(i) for i in range(2,121)]]
    #Y = dataset['Ret_MinusZero']

    #X['Feature_5'] = (X['Feature_5'] - np.mean(X['Feature_5']))/np.std(X['Feature_5'])
    #X['Feature_7'] = (X['Feature_7'] - np.mean(X['Feature_7']))/np.std(X['Feature_7'])

    # Part 1
#    model = Sequential()
#    model.add(Dense(34, input_dim=34))
#    model.add(Dense(34))
#    model.add(Dense(34))
#    model.add(Dense(1))
#    model.add(Activation('relu'))
#    model.compile(loss='mean_squared_error', optimizer='adam')
#
#    earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
#    history = model.fit(X.values, Y.values, batch_size=100, nb_epoch=100, validation_split=.15, callbacks=[earlystopper])
#    print("Done training " + str(datetime.datetime.now()))

    # Part 2
    model = Sequential()
    model.add(Dense(34, input_dim=34, activation='tanh'))
    #model.add(Activation('tanh'))
    model.add(Dense(250, activation='tanh'))
    #model.add(Activation('tanh'))
    #model.add(Dense(250))
    #model.add(Activation('tanh'))
    model.add(Dropout(0.5))    
    model.add(Dense(50))
    model.add(Activation('tanh'))
    model.add(Dense(1)) #, activation="relu"))
    model.add(Activation('tanh'))
    #model.add(Activation('relu'))
    model.compile(loss='mean_squared_error', optimizer='adam')

    earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    history = model.fit(X_train.values, Y_train.values, batch_size=100, nb_epoch=100, validation_split=.15, callbacks=[earlystopper])
    print("Done training " + str(datetime.datetime.now()))
    score = model.evaluate(X_test.values, Y_test.values, batch_size=100) 
    print('Problem 3 part 2a Test score : {}'.format(score))
    Y_predict = model.predict(X_test.values, batch_size=100)
    score = r2_score(Y_test.values, Y_predict)
    print('Problem 3 part 2b Test score : {}'.format(score))
    