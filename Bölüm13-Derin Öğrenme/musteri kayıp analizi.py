# -*- coding: utf-8 -*-
 # -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('Churn_Modelling.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)

#veri on isleme

X= veriler.iloc[:,3:13].values
Y = veriler.iloc[:,13].values



#encoder: Kategorik -> Numeric
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
X[:,1] = le.fit_transform(X[:,1])

le2 = preprocessing.LabelEncoder()
X[:,2] = le2.fit_transform(X[:,2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])],
                        remainder="passthrough"
                        )
X = ohe.fit_transform(X)
X = X[:,1:]




#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

# yapay sinir ağı oluşturma
import keras
from keras.models import Sequential
from keras.layers import Dense# nöron nesnesi oluşturacaz

classifier=Sequential()
classifier.add(Dense(6,init="uniform",activation="relu",input_dim=11,)) #  11 tane bağımsız değisken olduğu ıcın 11 verdimyapay sinir ağı katmanı ekleyeceğiz densenin içine kaç tane katman vereceğimizi yazacağız (gizli katman)

classifier.add(Dense(6,init="uniform",activation="relu"))

classifier.add(Dense(1,init="uniform",activation="sigmoid"))# çıkış katmanını 1 verdik
# bu kısma kadar ysa mızı tamamladık

classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

classifier.fit(X_train,y_train,epochs=50)

y_pred=classifier.predict(X_test) #burada X_testi makineye veriyoruz ama y_testi vermiyoruz bakalım veriler birbirini tutacakmı?

y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)




