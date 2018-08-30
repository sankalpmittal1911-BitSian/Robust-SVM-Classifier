import numpy as np 
import random
import warnings
import math
import csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_svmlight_file
from sklearn.svm import SVC
from sklearn import datasets, metrics, svm
from sklearn.metrics import (accuracy_score, average_precision_score,recall_score,classification_report)
from sklearn.model_selection import train_test_split
import pandas as pd
 
np.warnings.filterwarnings('ignore')
data=pd.read_csv('OnlineNewsPopularity.csv')
t = data.target
x = data.drop(['timedelta','url','shares','target'], axis=1)
 
#Training Data: 70% , Testing Data: 30%
x_train, x_test, t_train, t_test = train_test_split(x, t,test_size=0.3)
 
#Adding labeled noise to dataset
skel=list(range(0,len(t_train)))
random.shuffle(skel)
to_flip=skel[0:int(0.30*len(t_train))]
for i in to_flip:
	if(t_train.iloc[i]==0):
		t_train.iloc[i]=1
	else:
		t_train.iloc[i]=0
 
#Avoiding Funny looking rank 1 arrays
t_train=t_train.reshape(1,t_train.shape[0])
t_test=t_test.reshape(1,t_test.shape[0])
for i in range(t_train.shape[0]):
	t_train[0][i]=int(t_train[0][i])
for i in range(t_test.shape[0]):
	t_test[0][i]=int(t_test[0][i])
t_train[t_train==0]=-1
t_test[t_test==0]=-1
x_train=x_train.T
x_test=x_test.T
 
#Calculating c
 
R=np.linalg.norm(x_train)
c=(R**2)/(t_train.shape[1])
gamma=1/math.sqrt(2*c)
 
scaler = MinMaxScaler()
scaler.fit(x_train)
MinMaxScaler(copy=True, feature_range=(0, 1))
scaler.transform(x_train)
#gamma=0.01
lambd=0.5
 
def updateG_alpha_beta(param,w):
	return (1/x_train.shape[1])*(1/1-np.exp(-eta))*eta*(1-t_train*np.dot(w.T,x_train))*np.exp(-eta*param*(1-t_train*np.dot(w.T,x_train)))
 
def updateG_w(w):
	return (1/x_train.shape[1])*(1/(1-math.exp(-eta)))*eta*t_train*np.exp(-eta*alpha*(1-t_train*np.dot(w.T,x_train)))
 
def PdProx(param1,param2):
	param1=param2+gamma*G_alpha_beta
	param1[param1>1]=1
	param1[param1<0]=0
	return param1
 
def FISTA(param1,param2,param3,param4):
	temp2=(gamma*lambd*abs(param1)+0.5*(param1-(param2-gamma*G_w_alpha))**2)
	for j in range(w.shape[0]):
		if(temp2[j][0]<=param3[j][0]):
			param1[j][0]=temp2[j][0]
		else:
			param1[j][0]=param3[j][0]
	param3=gamma*lambd*abs(param1)+0.5*(param1-(param2-gamma*G_w_alpha)**2)
	temp=param4
	param4=(1+math.sqrt(1+4*(temp**2)))/2
	param2=param1+((temp-1)/param4) + (param1-param2)
	return param1,param2,param3,param4
 
	
 
 
#Random initialization of w and beta
w=np.zeros((x_train.shape[0],1))
beta=np.zeros((1,x_train.shape[1]))
alpha=np.zeros((1,x_train.shape[1]))
eta=1.0
 
G_alpha_alpha=updateG_alpha_beta(alpha,w)
G_alpha_beta=updateG_alpha_beta(beta,w)
G_w_alpha=updateG_w(w)
 
temp1=alpha*x_train
temp1=temp1.T
G_w_alpha=np.dot(G_w_alpha,temp1).T
G_alpha_alpha=G_alpha_alpha.T
G_alpha_beta=G_alpha_beta.T
alpha=alpha.T
beta=beta.T
 
#Initialization for FISTA
y=w
t=1
p=gamma*lambd*abs(w)+0.5*(w-(y-gamma*G_w_alpha)**2)
#num_iter=int(input("Enter Number of Iterations:"))
 
for i in range(1,5000):
	np.warnings.filterwarnings('ignore')
	
	#Running PdProx
	alpha=PdProx(alpha,beta)
	
	#Running FISTA
	w,y,p,t=FISTA(w,y,p,t)
	#Running PdProx
	beta=PdProx(beta,alpha)
	
	G_alpha_alpha=(1/x_train.shape[1])*(1/1-np.exp(-eta))*eta*(1-t_train*np.dot(w.T,x_train))*np.exp(-eta*alpha.T*(1-t_train*np.dot(w.T,x_train)))
	G_alpha_beta=(1/x_train.shape[1])*(1/1-np.exp(-eta))*eta*(1-t_train*np.dot(w.T,x_train))*np.exp(-eta*beta.T*(1-t_train*np.dot(w.T,x_train)))
	G_w_alpha=(1/x_train.shape[1])*(1/(1-math.exp(eta)))*eta*t_train*np.exp(-eta*alpha.T*(1-t_train*np.dot(w.T,x_train)))
	temp1=alpha.T*x_train
	temp1=temp1.T
	G_w_alpha=np.dot(G_w_alpha,temp1).T
	G_alpha_alpha=G_alpha_alpha.T
	G_alpha_beta=G_alpha_beta.T
 
correct=0
incorrect=0
 
t_predict=np.dot(w.T,x_test)
test=t_predict*t_test
for i in range(test.shape[1]):
	if(test[0][i]>=1):
		correct=correct+1
	elif(test[0][i]<=-1):
		incorrect=incorrect+1
	
accuracy=(correct/(correct+incorrect))*100
print("Accuracy is "+str(accuracy))
