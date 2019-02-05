# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 20:09:40 2018

@author: AKSPAN12
"""

#import library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

#using dendrogram to find optimal no. the clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))

#fitting clustering to dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

#plotting the clusters on graph
plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s = 100,c='red',label = 'Careful')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s = 100,c='blue',label = 'Standard')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s = 100,c='green',label = 'Target')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s = 100,c='cyan',label = 'Careless')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s = 100,c='magenta',label = 'Sensible')
plt.title('Clusters of client')
plt.xlabel('Annual income(K$)')
plt.ylabel('Spending score(1-100)')
plt.legend()
plt.show()