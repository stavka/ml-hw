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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

runProblem1 = False
runProblem2 = False
runProblem3 = True

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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


if(runProblem3):
    from keras.models import Sequential
    from keras.layers.core import Activation, Dense, Dropout, Merge
    from keras.callbacks import EarlyStopping

    featureModel = Sequential()
    featureModel.add(Dense(6, input_dim=6, activation='tanh'))
    #featureModel.add(Dense(32, activation='tanh'))
    featureModel.add(Dense(16, activation='tanh'))
    featureModel.add(Dense(6, activation='tanh'))

    volumeModel = Sequential()
    volumeModel.add(Dense(4, input_dim=4, activation='tanh'))
    #volumeModel.add(Dense(32, activation='tanh'))
    volumeModel.add(Dense(16, activation='tanh'))
    volumeModel.add(Dense(1, activation='tanh'))
    
    spreadModel = Sequential()
    spreadModel.add(Dense(4, input_dim=4, activation='tanh'))
    #spreadModel.add(Dense(32, activation='tanh'))
    spreadModel.add(Dense(16, activation='tanh'))
    spreadModel.add(Dense(1, activation='tanh'))
    
    askModel = Sequential()
    askModel.add(Dense(4, input_dim=4, activation='tanh'))
    #askModel.add(Dense(32, activation='tanh'))
    askModel.add(Dense(16, activation='tanh'))
    askModel.add(Dense(1, activation='tanh'))
    
    bidModel = Sequential()
    bidModel.add(Dense(4, input_dim=4, activation='tanh'))
    #bidModel.add(Dense(32, activation='tanh'))
    bidModel.add(Dense(16, activation='tanh'))
    bidModel.add(Dense(1, activation='tanh'))    
    
    avModel = Sequential()
    avModel.add(Dense(4, input_dim=4, activation='tanh'))
    #avModel.add(Dense(32, activation='tanh'))
    avModel.add(Dense(16, activation='tanh'))
    avModel.add(Dense(1, activation='tanh'))
    
    rvModel = Sequential()
    rvModel.add(Dense(4, input_dim=4, activation='tanh'))
    #rvModel.add(Dense(32, activation='tanh'))
    rvModel.add(Dense(16, activation='tanh'))
    rvModel.add(Dense(1, activation='tanh'))
    
    
    
    model = Sequential()
    model.add(Merge([featureModel, 
                     volumeModel,
                     spreadModel,
                     askModel,
                     bidModel,
                     avModel,
                     rvModel ], mode='concat', concat_axis=1))
                     
    model.add(Dense(64, activation='tanh'))
    model.add(Dense(1, activation='tanh'))
    
    
    model.compile(loss='mean_squared_error', optimizer='sgd')

    earlystopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    
    history = model.fit( [ X_train[['OrderQty','f1','f2','f3','time','volEstimate']].values,
                           X_train[['volume4','volume3','volume2','volume1']].values,
                           X_train[['spread4','spread3','spread2','spread1']].values,
                           X_train[['avgAskQty4','avgAskQty3','avgAskQty2','avgAskQty1']].values,
                           X_train[['avgBidQty4','avgBidQty3','avgBidQty2','avgBidQty1']].values,
                           X_train[['abnormalVolume4','abnormalVolume3','abnormalVolume2','abnormalVolume1']].values,
                           X_train[['residVol4','residVol3','residVol2','residVol1']].values ],
                         Y_train.values,
                         batch_size=100,
                         nb_epoch=100,
                         validation_split=.15,
                         callbacks=[earlystopper])
    
    
    
    
    
    
    print("Done training " + str(datetime.datetime.now()))
    score = model.evaluate([ X_test[['OrderQty','f1','f2','f3','time','volEstimate']].values,
                             X_test[['volume4','volume3','volume2','volume1']].values,
                             X_test[['spread4','spread3','spread2','spread1']].values,
                             X_test[['avgAskQty4','avgAskQty3','avgAskQty2','avgAskQty1']].values,
                             X_test[['avgBidQty4','avgBidQty3','avgBidQty2','avgBidQty1']].values,
                             X_test[['abnormalVolume4','abnormalVolume3','abnormalVolume2','abnormalVolume1']].values,
                             X_test[['residVol4','residVol3','residVol2','residVol1']].values ],
                          Y_test.values, batch_size=100) 
    print('Problem 3 part 2a Test score : {}'.format(score))
    Y_predict = model.predict([ X_test[['OrderQty','f1','f2','f3','time','volEstimate']].values,
                                X_test[['volume4','volume3','volume2','volume1']].values,
                                X_test[['spread4','spread3','spread2','spread1']].values,
                                X_test[['avgAskQty4','avgAskQty3','avgAskQty2','avgAskQty1']].values,
                                X_test[['avgBidQty4','avgBidQty3','avgBidQty2','avgBidQty1']].values,
                                X_test[['abnormalVolume4','abnormalVolume3','abnormalVolume2','abnormalVolume1']].values,
                                X_test[['residVol4','residVol3','residVol2','residVol1']].values ],
                              batch_size=100)
    score = r2_score(Y_test.values, Y_predict)
    print('Problem 3 part 2b Test score : {}'.format(score))
    