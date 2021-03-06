__author__ = 'YJY-1997'
#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

np.set_printoptions( suppress=True,)#linewidth=1000


dataMat = np.array(pd.read_excel('D://实验样本集.xls'))
y=dataMat[:,0]
X =dataMat[:,range(1,18,1)]
X_train0, X_test0, y_train0, y_test0 = train_test_split(X,y, test_size=0.9, random_state =133)#random_state = 1
skf=StratifiedKFold(n_splits=10,random_state=1,shuffle=True)
clf = LDA(n_components=None, priors=None, shrinkage=None,solver='eigen',store_covariance=False, tol=0.0001)#[0.8,0.2]

ACC=[]
F1=[]
TP=[]
FN=[]
FP=[]
TN=[]
PRECISION=[]
RECALL=[]
Max_0=[]
Min_1=[]

for k, (train, test) in enumerate(skf.split(X_train0,y_train0)):       #利用 模型划分数据集和目标变量 为一一对应的下标
    X_train,X_test=X_train0[train,],X_train0[test,]
    y_train,y_test=y_train0[train,],y_train0[test,]
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
    X_train_lda=clf.fit_transform(X_train_std, y_train)
    X_test_lda=clf.transform(X_test_std)
    y_predict=clf.predict(X_test_std)

    purple=[]
    yellow=[]
    for i in range(len(y_predict)):
        if y_predict[i]==1 and y_test[i]==0:
            purple.append(X_test_lda[i])
        if y_predict[i]==1 and y_test[i]==1:
            yellow.append(X_test_lda[i])
    if len(purple)!=0:
        max0=max(purple) #正类未违约最大值
        Max_0.append(max0[0])
    else:
        Max_0.append(0)
    min1=min(yellow)  #正类违约最小值
    Min_1.append(min1[0])

print("MAX_0:",Max_0)
print("MIN_1:",Min_1)
Max_0.sort()
Min_1.sort()
print("MAX_0 after sorted:",Max_0)
print("MIN_1 after sorted:",Min_1)
MAX_mean=np.mean(Max_0)
print("MAX_mean:",MAX_mean)
print('\n')

delta=0.0
if Min_1[0]< MAX_mean:
    delta=Min_1[0]
else:
    delta=(Min_1[0]+MAX_mean)/2

mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(15,6))

# 根据delta这个阈值打印10幅图与原LDA模型10幅图进行对比，即文中图3.5
for k, (train, test) in enumerate(skf.split(X_train0,y_train0)):       #利用模型划分数据集和目标变量 为一一对应的下标
    sc = StandardScaler()
    X_train0_std = sc.fit_transform(X_train0)
    X_test0_std = sc.transform(X_test0)
    X_train0_lda = clf.fit_transform(X_train0_std, y_train0)
    X_test0_lda = clf.transform(X_test0_std)
    y_predict0 = [0 for _ in range(len(X_test0_lda))]
    # print(type(y_predict0),y_predict0)#y_predict.shape,
    for i in range(len(X_test0_lda)):
        if X_test0_lda[i] >= delta:
            y_predict0[i] = 1
        if X_test0_lda[i] < delta:
            y_predict0[i] = 0

    k1=k+1
    if k1<=9:       #打印前9幅图
        plt.subplot(int('25{}'.format(k1)))
        plt.scatter(X_test_lda ,y_predict, marker='.',c=y_test)
        plt.title('图{} 第{}个模型预测结果'.format(k1,k1))
        plt.ylim([-0.05,1.05])
        plt.xlabel('X_test_LDA')
        plt.ylabel('y_predict')
    else:           #另一个窗口打印第10幅图，若导致前9幅图重叠，建议分两次打印，即分两次执行程序
        plt.figure(figsize=(3,3))
        plt.scatter(X_test_lda ,y_predict, marker='.',c=y_test)
        plt.title('图{} 第{}个模型预测结果'.format(k1,k1))
        plt.ylim([-0.05,1.05])
        plt.xlabel('X_test_LDA')
        plt.ylabel('y_predict')
plt.tight_layout()
plt.show()


#得到对比图后，使用改进LDA算法对整个数据集 (不再进行K折交叉验证) 进行训练，绘图查看整体分类效果，最终结果得到表3.6
sc = StandardScaler()
X_train0_std = sc.fit_transform(X_train0)
X_test0_std = sc.transform(X_test0)
X_train0_lda = clf.fit_transform(X_train0_std, y_train0)
X_test0_lda = clf.transform(X_test0_std)
y_predict0 = [0 for _ in range(len(X_test0_lda))]
# print(type(y_predict0),y_predict0)#y_predict.shape,
for i in range(len(X_test0_lda)):
    if X_test0_lda[i] >= delta:
        y_predict0[i] = 1
    if X_test0_lda[i] < delta:
        y_predict0[i] = 0

num = 0
tp, fn, fp, tn = 0, 0, 0, 0
for i in range(len(y_predict)):
    if y_predict[i] != y_test[i]:
        num += 1
    if y_test[i] == 1 and y_predict[i] == 1:
        tp += 1
    if y_test[i] == 1 and y_predict[i] == 0:
        fn += 1
    if y_test[i] == 0 and y_predict[i] == 1:
        fp += 1
    if y_test[i] == 0 and y_predict[i] == 0:
        tn += 1

p = tp / (tp + fp)
r = tp / (tp + fn)
f1 = 2 * tp / (2 * tp + fp + fn)
acc = accuracy_score(y_test, y_predict)
TP.append(tp)
FP.append(fp)
TN.append(tn)
FN.append(fn)
ACC.append(acc)
PRECISION.append(p)
RECALL.append(r)
F1.append(f1)
print('Misclassified samples: %d' % num)
print('Accuracy：%.8f' % accuracy_score(y_test, y_predict))
print("confusion_matrix:\n", metrics.confusion_matrix(y_test, y_predict))

fp_means=np.mean(FP)
fn_means=np.mean(FN)
tp_means=np.mean(TP)
tn_means=np.mean(TN)
acc_means=np.mean(ACC)
f1_means=np.mean(F1)
print('TP: ',tp_means)
print('FN: ',fn_means)
print('FP: ',fp_means)
print('TN: ',tn_means)
print(PRECISION)
print('PRECISION: ',np.mean(PRECISION))
print('RECALL: ',np.mean(RECALL))
print('ACC: ',acc_means)
print('F1: ',f1_means)
print('\n')

plt.figure(figsize=(4,4))
plt.scatter(X_test0_lda ,y_predict0, marker='.',c=y_test0)
plt.title('改进LDA图')
plt.ylim([-0.10,1.10])
plt.xlabel('X_test_LDA')
plt.ylabel('y_predict')
plt.tight_layout()
plt.show()
