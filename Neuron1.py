# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 14:26:41 2017

@author: Raja Ayyanar
"""

import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(-1, 1, 21)
y=1+2 *(x**2)
train_dataset=np.array([x,y])
train_dataset=np.transpose(train_dataset)
del x,y

no_of_inputs=2
no_of_hidden_neurons=5
no_of_output=1
MaxItertaion=200
del_v=np.random.uniform(0,1,(no_of_output,no_of_hidden_neurons))
del_v=np.transpose(del_v)
del_w=np.random.uniform(0,1,(no_of_hidden_neurons,no_of_inputs))
gamma_g=0.7
gamma_m=0.2
total_cycle=2000

x1=train_dataset[:,0]
x0=train_dataset[:,0]**0
all_x=np.array([x0 , x1])
all_x=np.transpose(all_x)
target_y=train_dataset[:,1]

w=np.random.uniform(0,1,(no_of_hidden_neurons,no_of_inputs))
v=np.random.uniform(0,1,(no_of_output,no_of_hidden_neurons))

for cycle in range(0,total_cycle):
    
    j=np.random.randint(0,np.size(target_y))
    one_x=np.array([all_x[j,:]]);
    a=w @ one_x.transpose();
    d=1/(1+np.exp(-a));
    y_cap=v @ d;
    
    error_y=target_y[j]- y_cap;
    error_d=v.transpose() @ error_y;
    error_a=(d*(1-d))*error_d
    del_v=gamma_g * d*error_y + gamma_m * del_v
    del_w=gamma_g * ( error_a @ one_x)  +gamma_m* del_w
    v=v+del_v.transpose()
    w=w+del_w
    
print('Final weights are',w)
    
#Testing

x_test=np.linspace(-1, 1, 201)
y_test=1+2*x_test*x_test
x0_test=x_test**0
testdata=np.array([x0_test, x_test, y_test])
testdata=testdata.transpose()

test_in=testdata[:,0:2]
e=np.zeros(max(np.shape(test_in)))
for test in range(0,max(np.shape(test_in))):
    one_x=np.array([test_in[test,:]])
    a=w@one_x.transpose();
    d=1/(1+np.exp(-a));
    y_cap=v@d;
    e[test]=y_cap-y_test[test]
    plt.plot(one_x[0,1],y_test[test],'ro')
    plt.hold(True)
    plt.plot(one_x[0,1],y_cap,'ko')
    plt.hold(True)



