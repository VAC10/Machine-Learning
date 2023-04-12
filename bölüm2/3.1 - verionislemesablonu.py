 # -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 18:50:13 2020

@author: sadievrenseker
"""

#1.kutuphaneler
import numpy as np
import matplotlib.pyplot as plt #görselleştirme için 
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('satislar.csv')
#pd.read_csv("veriler.csv")
#test
print(veriler)
#veri on isleme

aylar = veriler[['Aylar']]
print(aylar)

satislar = veriler[['Satislar']]
print(satislar)

satislar2 = veriler.iloc[:,:1].values
print(satislar2)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(aylar,satislar,test_size=0.33, random_state=0)
'''
#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train=sc.fit.transform(y_train)# y train le x traini yakın değerlere indirgedik 
Y_test=sc.fit.transform(y_test)# y train le x traini yakın değerlere indirgedik
'''
#Basit Doğrusal Regresyon Model(Model İnşaası) -Lineer regression
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train, y_train) # makinamız x trainden y traini öğrendi #fit fonksiyonuyla modeli inşaa edeceğiz burada x train y train diyerek yeni bir model inşaa et diyoruz. Aynı zamanda xtrainden y yi tahmin et dedik

tahmin=lr.predict(x_test )
x_train=x_train.sort_index()# yukarıda split fonksiyonunda random olarak indexleri atadığımız için veri görselleştirmemiz saçma gözüküyor bu yuzden bu metodla verileri sıraladım
y_train=y_train.sort_index()

plt.plot(x_train,y_train)
plt.plot(x_test,lr.predict(x_test))# "x texttteki her bir değer için o değerin karşılığı olan 48. satırda train ettiğimiz LinearRegression içindeki karşılıkları göster."demektir
plt.title("Aylara Göre Satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar") #metodun ustune gelip ctrl+i yaparak bilgi edinebilirsin




