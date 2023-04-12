# -*- coding: utf-8 -*-
#kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#kodlar
#veri yükleme
veriler=pd.read_csv("Eksik Veriler.csv ")

print(veriler)


#veri ön işleme
boy=veriler[["boy"]]
print (boy)

boyKilo=veriler[["boy","kilo"]]
print(boyKilo )

class insan:
    boy=180
    def kosmak(self,b):
        return b+10
ali=insan()
print(ali.boy)  
print(ali.kosmak(90))  
#eksik veriler düzeltme

#sci- kit learn 

from sklearn.impute import SimpleImputer #eksik verilerde NaN olan veriler tamamlamak için ordaki kolonun ortalamasını aldık ve bu kutuphane ile boş kısımlara yerleştireceğiz

imputer=SimpleImputer(missing_values=np.nan,strategy='mean')# strategyi mean olarak belirledik mean demek ortalamasını al demek
Yas=veriler.iloc[:,1:4].values
print (Yas)
imputer= imputer.fit(Yas[:,1:4])   #imputer objesinin fit fonksiyonunu çağırdık. fit fonksiyonu oğrenilecek olan değeri ifade eder
Yas[:,1:4]=imputer.transform(Yas[:,1:4]) # transformla nan değerlerini ortalama ya çevirecek yani transformla uygulamasını istedik.
print(Yas)





