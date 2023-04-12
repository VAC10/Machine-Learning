# -*- coding: utf-8 -*-
#1-kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#kodlar
#VERİ YÜKLEME
veriler=pd.read_csv("odev_tenis.csv ")

print(veriler)


#2-VERİ ÖN İŞLEME
#2.1 veri yükleme



#EKSİK VERİLERİN DÜZELTİLMESİ
#sci- kit learn 

 



#KATEGORİK VERİ OLUŞTURMA

from sklearn import preprocessing
#39uncu satırdan 47 ci satıra kadar değerleri teker teker labelencoding yapmak yerine aşağıdaki metodla tüm kolonlar label encoding yapılır
veriler2=veriler.apply(preprocessing.LabelEncoder().fit_transform)

c=veriler2.iloc[:,:1] #veriler 2 yi yukarıda label encoding yapmıştık. bu satırda ise outlook kolonumu label encoding yaptım

from sklearn import preprocessing
ohe=preprocessing.OneHotEncoder()# one hot encoder ile kategorik verileri binary(ikilik sisteme dönüştürdük) çünkü makine binary sayıları anlar.
c=ohe.fit_transform(c).toarray() # tek bi adımda ön işlemedeki öğrenme sürecini ülke kolonundan öğrenecek. bu ülke kolonu bir önceki aşamada sayıya çevirdiğimiz ülke kolonu öğrenip daha sonra tranform edecek toarrray diyerek numparrray olarak sonucu alacağız
print(c)

havaDurumu=pd.DataFrame(data=c, index=range(14),columns=["o","r","s"]) #dataframe oluşturup yukarıda binarye dönüştürdüğümüz hava durumlarını artık tabloya döktük
sonveriler=pd.concat([havaDurumu,veriler.iloc[:,1:3]],axis=1) #kendi yaptığımız hava durumu tablosu ile orjinal tablo olan veriler tablosundan 1 ile 3. satır arasındakı kolonları aldık boylelikle "humidty"ve "tempearature kolonları geldi"
sonveriler=pd.concat([veriler2.iloc[:,-2:],sonveriler],axis=1 ) #burada veriler2'den son iki kolonu aldık windy ve play kolonlarını da ekledik
#40.SATIRLA BİRLİKTE VERİ ÖN İŞLEME(DATA PREPROCESSİNG) AŞAMAMIZ BİTMİŞTİR !!!

"""
play=veriler.iloc[:,-1:].values #play'ın olduğu sutunu labelcoding yaptık. tek iki değişken olduğu için one hot coding yapmadım.
print (play)

le=preprocessing.LabelEncoder()
play[:,-1]=le.fit_transform(veriler.iloc[:,-1])
print(play)

windy=veriler.iloc[:,3:4].values
print(windy) #windy kolonunu bastık.
"""




# verilerin eğitim ve test için bölünmesi

# burada humidty bağımlı değişkendir multiple regression modelini kullanacağım humidty dışındaki kolonlar bağımsız  değişkenlerdir ve ben bunları kullanacağım

from sklearn.model_selection import train_test_split
# AŞAĞIDAKİ SATIRDA HUMİDTY(Nem) KOLONUNU TAHMİN ETTİRECĞİZ
x_train, x_test,y_train, y_test=train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:] ,test_size=0.33,random_state=0) #verileri ayırdık,yani son kolon hariç bütün kolonları al! dedik. çunku son kolonum bağımlı değişkenim
# x_train ve y_train verinin %67 lik kısmını  oluşrturur.
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train, y_train)# x trainden y trani öğren  x trainle y train arasında makine öğrenmesi modülü kuracağız.

y_pred=regressor.predict(x_test) #tahmin yazıyoruz(x'in test olarak ayrılmış kısmını daha önce 81-82-83 satırlı kodlarda öğrettiğim makine öğrenme algoritmasına gçre tahmin et ve çıkan tahmin sonucunu y_pred içine yaz)



#Geri Eleme (Backward eliminate) etkisi az olanları eğer sistemden kaldırırsak daha iyi bir model kurmuş oluruz.
import statsmodels.api as sm
X=np.append(arr=np.ones((14,1)).astype(int),values=sonveriler.iloc[:,:-1 ]) # append komutuyla dizi ekledik. np ones diyerek 1 lerden oluşan bir dizi oluşturduk, boyutunu 22.1 verdik (matris oluşturduk), matrisin tipi integerdır, ve bu işlemleri veri dizisine uygulayacak

X_list=sonveriler.iloc[:,[0,1,2,3,4,5]].values 
X_list=np.array(X_list,dtype=float)
model=sm.OLS(sonveriler.iloc[:,-1:],X_list).fit() #bu satır oluştuduğumuz regresyon için istatiksel değerlerimizi çıkartmaya yarıyor.(humidty nin p-value değerlerini çıkardık)
print(model.summary()) #backward elimination'da amacımız en fazla p-value değerini elemekti.

sonveriler=sonveriler.iloc[:,1:]#1.kolonu kaldırdık çunku p-value değeri en buyuk
#backward eliminationdan sonraki görünüm aşağıdaki kodlardadır
import statsmodels.api as sm
X=np.append(arr=np.ones((14,1)).astype(int),values=sonveriler.iloc[:,:-1 ]) # append komutuyla dizi ekledik. np ones diyerek 1 lerden oluşan bir dizi oluşturduk, boyutunu 14,1 verdik (matris oluşturduk), matrisin tipi integerdır, ve bu işlemleri veri dizisine uygulayacak

X_list=sonveriler.iloc[:,[0,1,2,3,4]].values 
X_list=np.array(X_list,dtype=float)
model=sm.OLS(sonveriler.iloc[:,-1:],X_list).fit() #bu satır oluştuduğumuz regresyon için istatiksel değerlerimizi çıkartmaya yarıyor.(humidty nin p-value değerlerini çıkardık)
print(model.summary())

x_train=x_train.iloc[:,1:]
x_test=x_test.iloc[:,1:]
regressor.fit(x_train, y_train)# x trainden y trani öğren  x trainle y train arasında makine öğrenmesi modülü kuracağız.

y_pred=regressor.predict(x_test)



