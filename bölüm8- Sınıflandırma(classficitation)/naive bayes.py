# -*- coding: utf-8 -*-
"""naïve Bayes sınıflandırması olasılık ilkelerine göre tanımlanmış 
bir dizi hesaplama ile, sisteme sunulan verilerin sınıfını
 yani kategorisini tespit etmeyi amaçlar.

Naïve Bayes sınıflandırmasında sisteme belirli bir oranda öğretilmiş veri sunulur
 (Örn: 100 adet). Öğretim için sunulan verilerin mutlaka bir sınıfı/kategorisi
 bulunmalıdır. Öğretilmiş veriler üzerinde yapılan olasılık işlemleri ile, 
 sisteme sunulan yeni test verileri, daha önce elde edilmiş olasılık değerlerine
 göre işletvilir ve verilen test verisinin hangi kategoride olduğu
 tespit edilmeye çalışılır. Elbette öğretilmiş veri sayısı ne kadar çok ise,
 test verisinin gerçek kategorisini tespit etmek o kadar kesin olabilmektedir. """
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

#svm
"""
Destek Vektör Makineleri (Support Vector Machine) genellikle sınıflandırma 
problemlerinde kullanılan gözetimli öğrenme yöntemlerinden biridir
. Bir düzlem üzerine yerleştirilmiş noktaları ayırmak için bir doğru çizer.
 Bu doğrunun, iki sınıfının noktaları için de maksimum uzaklıkta olmasını amaçlar
"""
from sklearn.svm import SVC
svc=SVC(kernel="rbf")
svc.fit(X_train,y_train)# modelimizi eğittik ve svc ile doğrusal bir ayrımı bulacak
y_predSvc=svc.predict(X_test)

cm=confusion_matrix(y_test, y_predSvc)
print("SVC")
print(cm)
#naive bayes
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
y_predNaive=gnb.predict(X_test)

cm=confusion_matrix(y_test, y_predNaive )
print("GNB")
print(cm)
