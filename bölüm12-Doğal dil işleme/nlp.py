import numpy as np
import pandas as pd
yorumlar=pd.read_csv("Restaurant_Reviews.csv",error_bad_lines=False)

import re
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
nltk.download("stopwords")
from nltk.corpus import stopwords
#preprocessing(ön işleme)
derlem=[]
for i in range(716): # tüm satırları donecegiz
    yorum=re.sub("[^a-zA-Z]"," ",yorumlar["Review"][i])#< ilk kolonda sadece harfleri al dedik. noktayı virgülü yok saydık
    # buyuk kucuk harf problemini cözme...
    yorum=yorum.lower()# tüm harfleri kucuk harfe cevirdik.
    yorum=yorum.split() # list haline cevirdik
    yorum=[ps.stem(kelime) for kelime in yorum if not kelime in set(stopwords.words("english"))]# kelimelerin icinde gezecez stop word olanları atacak
    yorum=''.join(yorum)
    derlem.append(yorum)
# su kısma kadar preprocessing yaptık yani veri ön işlemeyi yaptık

# feature exctraction(oznitelik cıkarımı)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=100) # countvectorizer nesnesin oluşturduk max_features ile en fazla kullanılan 200 kelimeyi al dedik
X=cv.fit_transform(derlem).toarray() # bagımsız degisken
y=yorumlar.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train ,X_test, y_train, y_test=train_test_split(X,y,test_size=0.33,random_state=0)


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
y_pred=gnb.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)


















