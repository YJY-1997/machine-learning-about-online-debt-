__author__ = 'YJY-1997'
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
np.set_printoptions(linewidth=1000) #suppress=True,

dataMat = np.array(pd.read_excel('D://实验样本集.xls'))
y1=dataMat[:,0]
X1 =dataMat[:,range(1,18,1)]
p = PCA(n_components=0.98)
main_vector = p.fit_transform(X1)
print(main_vector.shape)
x=main_vector[:,0]
y=main_vector[:,1]
plt.scatter(x,y, marker='.',alpha=0.5)
plt.grid(True)
plt.show()
