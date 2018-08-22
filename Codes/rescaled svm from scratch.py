import numpy as np 
import random
import warnings
import math
import csv
from sklearn.preprocessing import MinMaxScaler
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
to_flip=skel[0:int(0.3*len(t_train))]
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

scaler = MinMaxScaler()
scaler.fit(x_train)
MinMaxScaler(copy=True, feature_range=(0, 1))
scaler.transform(x_train)
gamma=0.01
lambd=0.5

#Random initialization of w and beta
w=np.zeros((x_train.shape[0],1))
beta=np.zeros((1,x_train.shape[1]))
alpha=np.zeros((1,x_train.shape[1]))
eta=0.2
G_alpha_alpha=(1/x_train.shape[1])*(1/1-np.exp(-eta))*eta*(1-t_train*np.dot(w.T,x_train))*np.exp(-eta*alpha*(1-t_train*np.dot(w.T,x_train)))
G_alpha_beta=(1/x_train.shape[1])*(1/1-np.exp(-eta))*eta*(1-t_train*np.dot(w.T,x_train))*np.exp(-eta*beta*(1-t_train*np.dot(w.T,x_train)))
G_w_alpha=(1/x_train.shape[1])*(1/(1-math.exp(-eta)))*eta*t_train*np.exp(-eta*alpha*(1-t_train*np.dot(w.T,x_train)))
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
	(alpha)=(beta)+gamma*G_alpha_beta
	(alpha)[(alpha)>1]=1
	(alpha)[(alpha)<0]=0
	
	#Running FISTA
	temp2=(gamma*lambd*abs(w)+0.5*(w-(y-gamma*G_w_alpha))**2)
	for j in range(w.shape[0]):
		if(temp2[j][0]<=p[j][0]):
			w[j][0]=temp2[j][0]
		else:
			w[j][0]=p[j][0]
	
	p=gamma*lambd*abs(w)+0.5*(w-(y-gamma*G_w_alpha)**2)
	temp=t
	t=(1+math.sqrt(1+4*(temp**2)))/2
	y=w+((temp-1)/t) + (w-y)
	#print(str(w.T)+"\n")
	beta=beta+gamma*G_alpha_alpha
	beta[beta>1]=1
	beta[beta<0]=0
	
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
