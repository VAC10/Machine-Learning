# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt #görselleştirme için 
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('veriler.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)
#veri on isleme

x=veriler.iloc[:,1:4].values#bağımsız değiken boy,kilo,yas aldık
y=veriler.iloc[:,4:].values#)bağımlı değişkenler
#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.33, random_state=0)

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


from sklearn.linear_model import LogisticRegression
logr=LogisticRegression(random_state=0)
logr.fit(X_train,y_train) # verimizi öğrendi

y_pred=logr.predict(X_test)
print(y_pred)
print(y_test)

#confusion matrix (karmasıklık matrisi)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
print(cm)# sınıflandırmanın dogrumu yanlısmı oldugunu tesbit ettik.

# KNN algoritması
# komsulara bakarak onlar arasındaki uzaklığa vs. bakarak sınıflandırma yapan algoritmadır
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1,metric='minkowski') #neighbor=kaç komsuya bakılacağı
knn.fit(X_train,y_train) # knni eğittim
y_predKnn=knn.predict(X_test)

cm=confusion_matrix(y_test, y_predKnn)
print(cm)









