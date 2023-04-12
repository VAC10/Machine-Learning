# -*- coding: utf-8 -*-
#kütüphaneler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#kodlar
#veri yükleme
veriler=pd.read_csv("veriler.csv ")

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