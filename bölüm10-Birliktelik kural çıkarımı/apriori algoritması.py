# -*- coding: utf-8 -*-
"""Apriori, genellikle market-sepet analizinde kullanılan bir algoritma. 
Her bir alışveriş bir transection olarak değerlendiriliyor.
 Her transection'ın içerdiği ürünlere göre; hangi ürünlerin birlikte alındığını
 belirlemek için kullanılıyor"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
veriler=pd.read_csv("sepet.csv",header=None)

t=[]
for i in range(0,7501):
    t.append([str(veriler.values[i,j]) for j in range(0,20)]) # veriler içindeki her bir satır ve sutunu okuduk

from apyori import apriori
kurallar=apriori(t,min_support=0.01,min_confidence=0.2,min_lift=3,min_length=2)
print(list(kurallar)) 


