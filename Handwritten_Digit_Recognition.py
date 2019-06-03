import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
path = os.getcwd() + '/Data/Handwritten_Digit_Training_Data.csv'
data = pd.read_csv(path, header=None)
data.head()
data.shape

cols = data.shape[1]
print("Columns = {}".format(cols))

y=np.array(data.iloc[:,:1])
x=np.array(data.iloc[:,1:cols])
x.shape, y.shape

np.unique(y)

rows = x.shape[0]

x = np.insert(x,0,values=np.ones(rows),axis=1)
x.shape

params = x.shape[1]
all_theta = np.zeros((4,params))
all_theta.shape

def sigmoid(z):
    return 1/(1+np.exp(-z))

print (sigmoid(0))
print (sigmoid(10))
print (sigmoid(100))

def computeCost(theta, x, y):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    
    first = np.multiply(-y,np.log(sigmoid(x*theta.T)))
    second = np.multiply((1-y),np.log(1-sigmoid(x*theta.T)))
    cost = np.sum(first-second)
    cost = cost/(2*len(x))
    return cost

cost = computeCost(all_theta, x, y)
print("Error = {}".format(cost))

learning_rate=1

def gradient(theta,x,y):
    theta = np.matrix(theta)
    x = np.matrix(x)
    y = np.matrix(y)
    error = sigmoid(x*theta.T)-y
    grad = (x.T*error)/(2*len(x))
    grad = grad*learning_rate
    return grad

from scipy.optimize import minimize

def one_vs_all(x, y, classes, new_theta):
    
    params = x.shape[1]
    rows = x.shape[0]
    
    for i in range(0, classes):
        theta = np.zeros(params)
        theta = new_theta[i,:]
        
        y_i = np.array([1 if label ==i else 0 for label in y])
        y_i = np.reshape(y_i, (rows, 1))
        
        fmin = minimize(fun=computeCost, x0=theta, args=(x, y_i), method='TNC', jac=gradient)
        
        all_theta[i,:] = fmin.x
        
    return all_theta

classes = 4
new_theta = one_vs_all(x, y, classes, all_theta)

computeCost(new_theta, x, y)

def predict_all(x,theta):
    x=np.matrix(x)
    theta=np.matrix(theta)
    
    h=sigmoid(x*theta.T)
    h_argmax=np.argmax(h,axis=1)
    
    return h_argmax

y_pred=predict_all(x, new_theta)
print(y_pred)

correct = [1 if a==b else 0 for (a,b) in zip(y_pred,y)]
print(correct)
accuracy=(sum(map(int,correct))/float(len(correct)))
print("Accuracy = {}%".format(accuracy*100))

import matplotlib.image as mpimg

def rgb2gray(rgb):
    return np.dot(rgb[:,:,:3],[0.299,0.587,0.114])

img=mpimg.imread('Data/Test.png')
gray=rgb2gray(img)
plt.imshow(gray,cmap=plt.get_cmap('gray'))

grarray=np.asarray(gray)
bw=(grarray<(grarray.mean()))*255
plt.imshow(np.reshape(bw,(28,28)),cmap=plt.cm.gray)

bw=bw.ravel()
bw=bw.reshape(1,-1)
brows=bw.shape[0]
bw=np.insert(bw,0,values=np.ones(brows),axis=1)

pred=predict_all(bw,new_theta)
if pred == [[0]]:
    digit = '0'
if pred == [[1]]:
    digit = '1'
if pred == [[2]]:
    digit = '2'
if pred == [[3]]:
    digit = '3'
print('The handwritten digit is', digit)