#Linear regression

import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Load the data
data = np.loadtxt("airfoil_self_noise.dat")

x = data[:,:5]
y = data[:,5]

sc  = StandardScaler()
x = sc.fit_transform(x)

x = np.concatenate((np.ones((len(x),1)),x),axis=1)

parms = np.array([.5]* 6)
parms

lr = .00001


# loss function
def get_loss(parms):
	yhat =  np.matmul(x,parms.T)
	mse = mean_squared_error(y,yhat)
	print(mse)
	
get_loss(parms)

for j in range(1000):
	p = np.zeros(6)
	for i in range(6):
		a= (np.matmul(x,parms) - y)
		p[i] = (lr/1503)*np.sum(a*x[:,i])
	
	get_loss(parms)
	parms = parms - p
	
	
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x,y)
yhat= lin_reg.predict(x)
mean_squared_error(y,yhat)

lin_reg.coef_
x = np.delete(x,np.s_[3],axis=1)

lin_reg.fit(x,y)
yhat= lin_reg.predict(x)
mean_squared_error(y,yhat)

#23.032747260592338

from matplotlib import pyplot as plt
plt.axis([0, 10, 0, 10])
for i in range(10):
    plt.scatter(i, i + 1)
    plt.pause(0.5)

plt.show()
