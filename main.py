import numpy as np
import pandas as pd
import pickle

data = pd.read_csv('Social_Network_Ads.csv')
print(data.head())

X= data.iloc[:,2:4].values #extract 2nd & 3rd colums
Y= data.iloc[:,-1].values #extract decision colums-(buy or not)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors =5)

print(knn.fit(X_train,y_train))

pickle.dump(knn,open('model.pkl','wb'))


## knn is an object of KNeighnorsClassifier class;
# This is the object that is trained and we will use this  to make predictions.
# But the problem is that this object is native to this file main.py.
# But we have to use this in a different file.
# Therfor pickel is use to searlize into a byte stream which will be stored in a seperate file.
# This file can be imported to any other file and then can be decerealized into a python object.

# The content of the pickel file depends on the model that is being used