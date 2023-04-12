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


#VERİ BÖLME İŞLEMİ


#Modelleme veya tahminlemede bulunmadan önce veri setimizi bir eğitim ve test setine ayırmalıyız. Böylelikle eğitim seti üzerinde modelimizi eğitir ve test seti üzerinde tahminler yapabiliriz.
#Train Set (Eğitim seti) : Modelin eğitildiği veri kümesidir.
#Test Set (Test seti) : Bir eğitim kümesinde geliştirilen modeli değerlendirmek için kullanılan bir veri kümesidir.
from sklearn.model_selection import train_test_split #veri bölme kütüphanemizi dahil ettik 
x_train,x_test,y_train,y_test=train_test_split(s,sonuc3, test_size=0.33,random_state=0)

#ÖZNİTELİK ÖLÇEKLEME (feature scaling)
#Türkçesi özellik ölçekleme olarak geçen feature scaling ile değişkenleri makine öğrenmesi algoritmalarına sokmadan önce belirlediğimiz aralıklara indirgiyoruz
from sklearn.preprocessing import StandardScaler #standart scaler kullanırak birbirine yakın halde ölçekledik
sc=StandardScaler()# farklı dunyalardaki verileri aynı dunyaya çektik

X_train=sc.fit_transform(x_train) #x_train'i dönüştürme işlemi yaptık parametreye x_train yazarak ss_traindeki verileri scaled yaptık
X_test=sc.fit_transform(x_test) #x_test'i ölçekledik













