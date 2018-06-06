# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 20:16:59 2018

@author: Raja Ayyanar
"""

def reluDerivative(x):
    
    x[x<=0] = 0
    x[x>0] = 1
    return x


import numpy as np
import matplotlib.pyplot as plt
import timeit
start = timeit.timeit()

#Creating Training Datasets
x1=np.random.uniform(-1, 1,1000)
x2=np.random.uniform(-1,1,1000)
y=x1**2 - x2**2
train_dataset=np.array([x1,x2,y])
train_dataset=np.transpose(train_dataset)
del x1,x2,y


#Layer H1 parameters
no_of_inputs_h1=3
no_of_hidden_neurons_h1=50
no_of_output_h1=no_of_hidden_neurons_h1
MaxItertaion=1000
del_v1=np.random.uniform(0,1,(no_of_output_h1,no_of_hidden_neurons_h1))
del_v1=np.transpose(del_v1)
del_w1=np.random.uniform(0,1,(no_of_hidden_neurons_h1,no_of_inputs_h1))
gamma_g_h1=0.8
gamma_m_h1=0.2


#Layer H2 Parameters
no_of_inputs_h2=no_of_output_h1 + 1
no_of_hidden_neurons_h2=30
no_of_output_h2=1
MaxItertaion=1000
del_v2=np.random.uniform(0,1,(no_of_output_h2,no_of_hidden_neurons_h2))
del_v2=np.transpose(del_v2)
del_w2=np.random.uniform(0,1,(no_of_hidden_neurons_h2,no_of_inputs_h2))
gamma_g_h2=0.8
gamma_m_h2=0.2
total_cycle=5000


# inputs for H1 
x1=train_dataset[:,0]
x2=train_dataset[:,1]
x0=train_dataset[:,0]**0

all_x=np.array([x0 , x1, x2])
all_x=np.transpose(all_x)
target_y=train_dataset[:,2]

#weight initialization
w1=np.random.uniform(0,1,(no_of_hidden_neurons_h1,no_of_inputs_h1))
v1=np.random.uniform(0,1,(no_of_output_h1,no_of_hidden_neurons_h1))
w2=np.random.uniform(0,1,(no_of_hidden_neurons_h2,no_of_inputs_h2))
v2=np.random.uniform(0,1,(no_of_output_h2,no_of_hidden_neurons_h2))

e=np.zeros(total_cycle)
ew1=np.zeros((total_cycle,no_of_hidden_neurons_h1,no_of_inputs_h1))

for cycle in range(0,total_cycle):
    
    j=np.random.randint(0,np.size(target_y))
    one_x=np.array([all_x[j,:]]);
    a=w1 @ one_x.transpose();
    d=np.maximum(0,a);  #ReLu Activation on H1
    bias1=np.ones((1,1))
    bd=np.concatenate((d,bias1), axis=0) #adding bias for layer2
    a2=w2@bd
    d2=np.maximum(0,a2)

    y_cap=v2 @ d2;
    
    error_y=target_y[j]- y_cap;

    error_d2=v2.transpose() @ error_y;
    error_a2=reluDerivative(d2)*error_d2
   
    error_d=w2.transpose() @ a2;
    error_a=reluDerivative(d)*error_d[:-1,:]

    del_v2=gamma_g_h2 * d2*error_y + gamma_m_h2 * del_v2
    #del_v1=gamma_g_h1 * d*a2 + gamma_m_h1 * del_v1
    bd2=np.concatenate((a2,bias1), axis=0) #backprop for bias
    del_w2=gamma_g_h1 * (a2@bd.T) + gamma_m_h2 * del_w2
    del_w1=gamma_g_h1 * ( error_a @ one_x)  +gamma_m_h1* del_w1
    v2=v2-del_v2.transpose()
    #v1=v1-del_v1.transpose()
    w2=w2-del_w2
    w1=w1-del_w1
    e[cycle]=error_y
    ew1[cycle,:,:]=w1
    
    
#print('Final weights are',w1,w2,v1)
    


#Testing
x_test=np.random.uniform(-1, 1,1000)
x2_test=np.random.uniform(-1,1,1000)
y_test=x_test**2 - x2_test**2
x0_test=x_test**0
testdata=np.array([x0_test, x_test,x2_test, y_test])
testdata=testdata.transpose()

test_in=testdata[:,0:3]
e_test=np.zeros(max(np.shape(test_in)))
for test in range(0,max(np.shape(test_in))):
    one_x=np.array([test_in[test,:]])
    a=w1 @ one_x.transpose();
    d=np.maximum(0,a);  #ReLu Activation on H1
    bias1=np.ones((1,1))
    bd=np.concatenate((d,bias1), axis=0) #adding bias for layer2
    a2=w2@bd
    d2=np.maximum(0,a2)

    y_cap=v2 @ d2;    
    e_test[test]=y_cap-y_test[test]

end = timeit.timeit()
print(np.mean(np.square(e_test)))
print(end-start)








