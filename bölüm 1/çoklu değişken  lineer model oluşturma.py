# -*- coding: utf-8 -*-
#1-kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#kodlar
#VERİ YÜKLEME
veriler=pd.read_csv("veriler.csv ")

print(veriler)


#2-VERİ ÖN İŞLEME
#2.1 veri yükleme



#EKSİK VERİLERİN DÜZELTİLMESİ
#sci- kit learn 

 



#KATEGORİK VERİ OLUŞTURMA
ulke=veriler.iloc[:,0:1].values #ülkelerimizin olduğu kolonları iloc fonksiyonu ile çağırdık.
print (ulke)
from sklearn import preprocessing
#
le= preprocessing.LabelEncoder()  #label encoder ile ülke isimlerini fit_transform fonkisyonuyla sayısal değere dönderdik.
#kategorik verinin sayısal veriye dönüştürülmesi
ulke[:,0]=le.fit_transform(veriler.iloc[:,0]) #[:,0] diyerek ilk kolonu transform ettik
print (ulke)
ohe=preprocessing.OneHotEncoder()# one hot encoder ile kategorik verileri binary(ikilik sisteme dönüştürdük) çünkü makine binary sayıları anlar.
ulke=ohe.fit_transform(ulke).toarray() # tek bi adımda ön işlemedeki öğrenme sürecini ülke kolonundan öğrenecek. bu ülke kolonu bir önceki aşamada sayıya çevirdiğimiz ülke kolonu öğrenip daha sonra tranform edecek toarrray diyerek numparrray olarak sonucu alacağız

print  (ulke)

Yas=veriler.iloc[:,1:4].values


c=veriler.iloc[:,-1:].values #cinsiyet olduğu kolonları iloc fonksiyonu ile çağırdık.
print (c)
from sklearn import preprocessing
#
le= preprocessing.LabelEncoder()  #label encoder ile ülke isimlerini fit_transform fonkisyonuyla sayısal değere dönderdik.
#kategorik verinin sayısal veriye dönüştürülmesi
c[:,-1]=le.fit_transform(veriler.iloc[:,-1]) #[:,0] diyerek ilk kolonu transform ettik
print (c)
ohe=preprocessing.OneHotEncoder()# one hot encoder ile kategorik verileri binary(ikilik sisteme dönüştürdük) çünkü makine binary sayıları anlar.
c=ohe.fit_transform(c).toarray() # tek bi adımda ön işlemedeki öğrenme sürecini ülke kolonundan öğrenecek. bu ülke kolonu bir önceki aşamada sayıya çevirdiğimiz ülke kolonu öğrenip daha sonra tranform edecek toarrray diyerek numparrray olarak sonucu alacağız

print  (c)
#data frame ile verilere index numaraları girilmesi ve sutun isimleri verilmesi (numpy dizileri dataframe dönüşümü)
sonuc=pd.DataFrame(data=ulke, index=range(22),columns=["fr","tr","us"])#dataframeler indexleri ve kolonları olan veri yapılarıdır 
print(sonuc)

sonuc2=pd.DataFrame(data=Yas,index=range(22),columns=["boy","kilo","cinsiyet"]) #bu seferde boy kilo cinsiyeti dataframe olarak aldık. 
print (sonuc2)

cinsiyet= veriler.iloc[:,- 1].values #cinsiyet verilerini ayırdık,söktük sondan birinci kolonda olduğu için -1 dedik
print (cinsiyet)


sonuc3=pd.DataFrame(data=c[:,:1], index=range(22),columns=["cinsiyet"])#dummy variable tuzağına düşmemek için cinsiyet tablosundaki iki sütünu tek sutuna indirdik
print(sonuc3)

s=pd.concat([sonuc,sonuc2], axis=1) #farklı dataframeleri alarak concate ettik yani birleştrip 's' değişkenine atadık
#axis e 1 vererek  yönü doğru verdik. alt alta değilde yan yana oluşturduk yani satır sayısını eşledik ve NaN atamasını engelledik
#iki veriyi birleştiriyoruz ama axis=1 komutu ile aynı olan satırları atıyoruz
print (s)

s2=pd.concat([s,sonuc3],axis=1)
print (s2)

# BU BÖLÜM İÇİN ÖNEMLİ KISIM
# verilerin eğitim ve test için bölünmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test=train_test_split(s,sonuc3, test_size=0.33,random_state=0) #burada test için %33'lük bir veri oluşturduk.
# x_train ve y_train verinin %67 lik kısmını  oluşrturur.
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train, y_train)# x trainden y trani öğren  x trainle y train arasında makine öğrenmesi modülü kuracağız.

y_pred=regressor.predict(x_test) #tahmin yazıyoruz(x'in test olarak ayrılmış kısmını daha önce 81-82-83 satırlı kodlarda öğrettiğim makine öğrenme algoritmasına gçre tahmin et ve çıkan tahmin sonucunu y_pred içine yaz)

#boy kolonunu çekip makine öğrenme algoritmamıza dahil edelim..
boy=s2.iloc[:,3:4]
print(boy)

sol=s2.iloc[:,:3] # butun satırları al ama 3 üncü kolondan öncekileri al.
sag=s2.iloc[:,4:]# butun satırları al ama 4. kolondan sonrakini al

veri=pd.concat([sol,sag],axis=1) # sol ve sağdaki verileri "veri" adlı değişkene atadık. artık veride boy gözükmeyecek
x_train, x_test,y_train, y_test=train_test_split(veri,boy, test_size=0.33,random_state=0)
r2=LinearRegression()
r2.fit(x_train, y_train)

y_pred=r2.predict(x_test) # x tessten y testi tahmin edecek.

#Geri Eleme (Backward eliminate)
import statsmodels.api as sm
X=np.append(arr=np.ones((22,1)).astype(int),values=veri) # append komutuyla dizi ekledik. np ones diyerek 1 lerden oluşan bir dizi oluşturduk, boyutunu 22.1 verdik (matris oluşturduk), matrisin tipi integerdır, ve bu işlemleri veri dizisine uygulayacak

X_list=veri.iloc[:,[0,1,2,3,4,5]].values 
X_list=np.array(X_list,dtype=float)
model=sm.OLS(boy,X_list).fit() #bu satır oluştuduğumuz regresyon için istatiksel değerlerimizi çıkartmaya yarıyor.
print(model.summary()) #backward elimination'da amacımız en fazla p-value değerini elemekti.

X_list=veri.iloc[:,[0,1,2,3,5]].values #burada en yuksek p-value değeri olan 4. elemanı eledik.
X_list=np.array(X_list,dtype=float)
model=sm.OLS(boy,X_list).fit() #bu satır istatiksel değerlerimizi çıkartmaya yarıyor.
print(model.summary()) #backward eliminationda amacımız en fazla p-value değerini elemekti.




