import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
from sklearn.cluster import AgglomerativeClustering 

X = np.array([[5,3], [10,15], [15,12], [24,10], [30,30], [85,70], [71,80], [60,78], [70,55], [80,91] ])


cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward') 
cluster.fit_predict(X)

plt.scatter(X[:,0],X[:,1], c=cluster.labels_, cmap='rainbow') 
plt.show()