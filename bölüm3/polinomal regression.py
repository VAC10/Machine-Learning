# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt #görselleştirme için 
import pandas as pd

#2.veri onisleme
#2.1.veri yukleme
veriler = pd.read_csv('maaslar.csv')

x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]


#linear regression
#doğrusal model oluşturma
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x,y)




#polynomial regression 
#doğrusal olmayan(non linear) model oluşturma
#2.dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(x)
lin_reg2=LinearRegression()
lin_reg2.fit(x_poly,y)

#4.dereceden polinom
poly_reg3=PolynomialFeatures(degree=4)
x_poly3=poly_reg3.fit_transform(x)
lin_reg3=LinearRegression()
lin_reg3.fit(x_poly3,y)

#görselleştirme
plt.scatter(x,y,color='red')
plt.plot(x,lin_reg.predict(x),color='purple')


plt.scatter(x,y,color="red")
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)),color='blue') # her bir datapoint için yani x için önce polynomal biçime dönüştür
plt.show()


plt.scatter(x,y,color="red")
plt.plot(x,lin_reg3.predict(poly_reg3.fit_transform(x)),color='blue') # her bir datapoint için yani x için önce polynomal biçime dönüştür
plt.show()


#tahminler

#Linear regression ile;
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]])) # buradaki değerin 16 tl çıkmasının sebebi linear regressionda değerlerin olduğundan yüksek verilmesidir.
 

#polynomal reg ile ;
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))

print(lin_reg2.predict(poly_reg.fit_transform([[11]])))# tahmini vermeden önce transform yaptık yani polynomal regressiona çevirdik







