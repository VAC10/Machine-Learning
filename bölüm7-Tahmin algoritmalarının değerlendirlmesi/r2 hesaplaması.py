# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt #görselleştirme için 
import pandas as pd
from sklearn.metrics import r2_score # tahminler arasında değerlednirme (r-kare ) kutuphanesi
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

print("Linear  R2 değeri")
print(r2_score(y,lin_reg.predict(x)))


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
print("Polynomial  R2 değeri")
print(r2_score(y,lin_reg2.predict(poly_reg.fit_transform(x))))


from sklearn.preprocessing import StandardScaler

sc1=StandardScaler() # scaling svr de önemlidir ! ! !
x_olcekli = sc1.fit_transform(x)

sc2=StandardScaler()
y_olcekli= sc2.fit_transform(y)

from sklearn.svm import SVR
svr_reg=SVR(kernel="rbf")
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color="red")
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color="blue") # her bir x değeri için tahminde bulun ve her bir x değeri için svr deki prediction karşılığını bul ve görselleştir
plt.show()
print(svr_reg.predict([[11]]))
print(svr_reg.predict([[6.6]]))
print("SVR R2 değeri")
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))

#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor

r_dt=DecisionTreeRegressor(random_state=0)
r_dt.fit(x,y) #x ten y yi öğren

plt.scatter(x,y,color="red")
plt.plot(x,r_dt.predict(x),color="yellow")
z=x+0.5
k=x-0.5

plt.plot(x,r_dt.predict(z),color="green" )
plt.plot(x,r_dt.predict(k),color="pink" )

plt.show()
print(r_dt.predict([[11]]))
print(r_dt.predict([[6.2]]))
print("Decision Tree R2 değeri")
print(r2_score(y,r_dt.predict(x)))

# random forest regression
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=10,random_state=0) # estimator: kaç tane decision tree çizileceğini ifade eder.
rf_reg.fit(x,y.values.ravel())
print(rf_reg.predict([[6.5]])) # 6.5 iş grubuna sahip kişinin maaşını tahmin et

plt.scatter(x,y,color="red")
plt.plot(x,rf_reg.predict(x),color="blue")

plt.plot(x,rf_reg.predict(z),color="green")
plt.show()


print("Random Forest R2 değeri")
print(r2_score(y,rf_reg.predict(x)))
print(r2_score(y,rf_reg.predict(k)))
print(r2_score(y,rf_reg.predict(z)))

# Ozet R2 degerleri
print("-----------------------------")

print("Linear  R2 değeri")
print(r2_score(y,lin_reg.predict(x)))


print("Polynomial  R2 değeri")
print(r2_score(y,lin_reg2.predict(poly_reg.fit_transform(x))))

print("Decision Tree R2 değeri")
print(r2_score(y,r_dt.predict(x)))


print("Random Forest R2 değeri")
print(r2_score(y,rf_reg.predict(x)))

print("SVR R2 değeri")
print(r2_score(y_olcekli,svr_reg.predict(x_olcekli)))
