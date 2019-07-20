## a simple regretion plrobplem using sklearning.linear models
## by sajjad p. savoji
## sat , 20 july 2019

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
import sklearn.neighbors as sn

#load the data
bli = pd.read_csv("BLI.csv" , thousands=',')
gdp = pd.read_csv('WEO_Data.csv' , thousands=',')

#prepare the data
countries_bli = bli.Country
countries_gdp = gdp.Country
countries_bli = [i for i in countries_bli]
countries_gdp = [i for i in countries_gdp]

bli = bli.Value
gdp = gdp['2015']
bli = [i for i in bli]
gdp = [i for i in gdp]

y =[]
x=[]
for i in range(len(bli)) :
    try :
        x.append( gdp[ countries_gdp.index( countries_bli[i])])
        y.append(bli[i])
    except:
        continue

#select a linear model
model = lm.LinearRegression()
n_model = sn.KNeighborsRegressor(n_neighbors= 3)
#train the model
x=[[i] for i in x]
model.fit(x,y)
n_model.fit(x,y)

#predict some data
x_pred = [ i for i in range(int(max(x)[0]))]
x_pred = [[i] for i in x_pred]
y_pred_l = model.predict(x_pred)
y_pred_n = n_model.predict(x_pred)
#plot the data
plt.scatter(x,y, color='gray' , label='data points')
plt.plot(x_pred , y_pred_l , color='blue' , linewidth='3' , label='linear regretion')
plt.plot(x_pred , y_pred_n , color='green' , linewidth='3' , label='kNeighbor regretion')
plt.ylabel('life satisfaction')
plt.xlabel('income')
plt.legend()
plt.show()