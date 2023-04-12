# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


data=pd.read_excel("Iris.xls")

x=data.iloc[:,1:4].values#bagımısız degiskenler
y=data.iloc[:,4:].values#bagımlı degisken

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

#veri olcekleme
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)

# buradan itibaren sınıflandırma algoritmaları baslar
# 1- Logistic Regression

from sklearn.linear_model import LogisticRegression
logr=LogisticRegression(random_state=0)
logr.fit(X_train, y_train)# eğitim
y_pred=logr.predict(X_test)
print("lR")
#karmasıklık matrisi
cm=confusion_matrix(y_test, y_pred)
print(cm)

# KNN algoritması
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1,metric="minkowski")
knn.fit(X_train,y_train) # algoritmayı egitiyoruz
y_predKnn=knn.predict(X_test) #tahmin
cm=confusion_matrix(y_test, y_predKnn)
print("KNN")
print(cm)


#SVC 
from sklearn.svm import SVC
svc=SVC(kernel="poly")
svc.fit(X_train,y_train)

y_predSvc=svc.predict(X_test)
cm=confusion_matrix(y_test, y_predSvc)
print("SVC")
print(cm)

#Naive bayes
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
y_predGnb=gnb.predict(X_test)
cm=confusion_matrix(y_test, y_predGnb)
print("Naive")
print(cm)

# Decision tree classifier
from sklearn.tree import DecisionTreeClassifier
dtc=DecisionTreeClassifier(criterion="entropy")
dtc.fit(X_train, y_train)
y_predDtc=dtc.predict(X_test)

cm=confusion_matrix(y_test, y_predDtc)
print("Decision Tree")
print(cm)

# Random forest
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=10,criterion="entropy")
rfc.fit(X_train,y_train)
y_predRfc=rfc.predict(X_test)

cm=confusion_matrix(y_test, y_predRfc)
print("Rfc")
print(cm)


#ROC ,TPR,FPR degerleri
from sklearn import metrics
y_proba=rfc.predict_proba(X_test) # sınıflandırma olasılıkları
print(y_test)
print(y_proba[:,0])
fpr, tpr ,thold=metrics.roc_curve(y_test,y_proba[:,0],pos_label='e')
print(fpr)
print(tpr) 

























