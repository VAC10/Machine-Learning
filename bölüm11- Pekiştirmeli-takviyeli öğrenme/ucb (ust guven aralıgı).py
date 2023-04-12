# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler=pd.read_csv("Ads_CTR_Optimisation.csv")
# random selection 
"""
import random
N=10000
d=10
toplam=0
secilenler=[]
for n in range(0,N):
    ad=random.randrange(d)
    secilenler.append(ad)
    odul=veriler.values[n,ad] # verilerdeki n.satır=1 ise ödül 1
    toplam=toplam+odul
    

plt.hist(secilenler)    
plt.show()

"""
import math
#ucb algoritması
N=10000 # 10000 islem
d=10
oduller=[0]*d# ilk basta butun ilanların odulu0
toplam=0# toplam odul
tiklamalar=[0]*d# o ana kadarki tıklamalar
secilenler=[]
for n in range(1,N):
    ad=0#secilen ilan
    max_ucb=0
    
    for i in range(0,d):
        if(tiklamalar[i]>0):
            ortalama=oduller[i]/tiklamalar[i]
            delta=math.sqrt(3/2*math.log(n)/tiklamalar[i])
            ucb=ortalama+delta
        else:
            ucb=N*10
        if max_ucb< ucb:# maxtan buyuk bir ucb cıktı
            max_ucb=ucb
            ad=i
    secilenler.append(ad)
    tiklamalar[ad]=tiklamalar[ad]+1         
    odul=veriler.values[n,ad]
    oduller[ad]=oduller[ad]+odul
    toplam=toplam+odul
    
print("toplam odul:")  
print(toplam)  
    
plt.hist(secilenler) 
plt.show()   
    
    
    
    
    
    
    
    
    
































































