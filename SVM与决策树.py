__author__ = 'YJY-1997'
#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import math
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import cross_val_score
np.set_printoptions( suppress=True,)#linewidth=1000

dataMat = np.array(pd.read_excel('D://实验样本集.xls'))
y=dataMat[:,0]
X =dataMat[:,range(1,18,1)]
ss = StandardScaler()
X_train1 = ss.fit_transform(X)
ACC=[]
F1=[]
TP=[]
FN=[]
FP=[]
TN=[]
PRECISION=[]
RECALL=[]
Fea_impor0=[]
Fea_impor1=[]
Fea_impor2=[]
Fea_impor3=[]
Fea_impor4=[]
Fea_impor5=[]
Fea_impor6=[]
Fea_impor8=[]
Fea_impor9=[]
Fea_impor7=[]
Fea_impor10=[]
Fea_impor11=[]
Fea_impor12=[]
Fea_impor13=[]
Fea_impor14=[]
Fea_impor15=[]
Fea_impor16=[]
skf=StratifiedKFold(n_splits=10,random_state=1,shuffle=True)
# skf=KFold(n_splits=10,random_state=1,shuffle=True)
for train_index,test_index in skf.split(X,y):
    # print("train\n",train_index,"\ntest\n",test_index)
    X_train,X_test=X[train_index,],X[test_index,]
    y_train,y_test=y[train_index,],y[test_index,]
    # print("train\n",y_train,"\ntest\n",y_test)

    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)

#LDA 线性判别分析
    # #clf = LDA(n_components=None, priors=None, shrinkage=None, solver='svd',store_covariance=False, tol=0.0001)
    # clf = LDA(n_components=None, priors=[0.95,0.05], shrinkage=None,solver='eigen',store_covariance=False, tol=0.0001)#priors=None
    # X_train_lda=clf.fit_transform(X_train_std, y_train)
    # X_test_lda=clf.transform(X_test_std)
#SVM
    # 线性核函数初始化
    clf = LinearSVC(loss='hinge',max_iter=10000)#LinearSVC（线性 SVM 算法）,最大迭代次数次
    # #clf =SVC(kernel='rbf', probability=True,random_state=1)   # SVC模型
    # clf =SVC(kernel='linear')#SVC（SVM 算法）,最大迭代次数max_iter=2000次
#线性回归
    # clf =LogisticRegression(max_iter=1000)
#决策树
    # clf = tree.DecisionTreeClassifier(criterion='entropy')#使用特征选择标准为entropy,默认为基尼系数”gini”
    # DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
    #                    max_features=None, max_leaf_nodes=None,
    #                    min_impurity_decrease=0.0, min_impurity_split=None,
    #                    min_samples_leaf=1, min_samples_split=2,
    #                    min_weight_fraction_leaf=0.0, presort=False,
    #                    random_state=None, splitter='best')

    clf.fit(X_train_std, y_train)
    y_predict=clf.predict(X_test_std)
    print(str(clf))

    num = 0
    tp,fn,fp,tn=0,0,0,0
    for i in range(len(y_predict)):
        if y_predict[i] != y_test[i]:
            num += 1
        if  y_test[i]==1 and y_predict[i]==1:
            tp+=1
        if  y_test[i]==1 and y_predict[i]==0:
            fn+=1
        if  y_test[i]==0 and y_predict[i]==1:
            fp+=1
        if  y_test[i]==0 and y_predict[i]==0:
            tn+=1

    p=tp/(tp+fp)
    r=tp/(tp+fn)
    f1=2*tp/(2*tp+fp+fn)
    acc=accuracy_score(y_test,y_predict)
    TP.append(tp)
    FP.append(fp)
    TN.append(tn)
    FN.append(fn)
    ACC.append(acc)
    PRECISION.append(p)
    RECALL.append(r)
    F1.append(f1)
    # print(tp)
    # print(fn)
    # print(fp)
    # print(tn,'\n')
    # print(p,)
    # print(r,)
    # print(f1,)
    # print('Misclassified samples: %d' % num)
    # print('Accuracy：%.8f'% acc)
    # print('The Accuracy of Linear SVC is', clf.score(X_test_std, y_test))
    print("confusion_matrix:\n",metrics.confusion_matrix(y_test,y_predict))
    print("precision_recall_f1-score_accuracy:\n",metrics.classification_report(y_test,y_predict))
    # print("协方差 ",clf.covariance_ )
    # print("比例 ",clf.explained_variance_ratio_)
    # print("Feature importances:",clf.feature_importances_)


    # c=clf.coef_[0];
    # print("系数：",c)
    # Fea_impor0.append(c[0])
    # Fea_impor1.append(c[1])
    # Fea_impor2.append(c[2])
    # Fea_impor3.append(c[3])
    # Fea_impor4.append(c[4])
    # Fea_impor5.append(c[5])
    # Fea_impor6.append(c[6])
    # Fea_impor7.append(c[7])
    # Fea_impor8.append(c[8])
    # Fea_impor9.append(c[9])
    # Fea_impor10.append(c[10])
    # Fea_impor11.append(c[11])
    # Fea_impor12.append(c[12])
    # Fea_impor13.append(c[13])
    # Fea_impor14.append(c[14])
    # Fea_impor15.append(c[15])
    # Fea_impor16.append(c[16])

    #获得线性函数的系数
    # print("support vectors:",clf.support_vectors_)
    #支持向量
    # print("position of SV:",clf.support_)
    #支持向量的位置
    # print("number of SV:",clf.n_support_)
    #支持向量的个数

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
a=cross_val_score(clf, X_train1, y, cv=10, scoring='roc_auc')
print("auc:",a)
print("MEAN AUC ",a.mean())

# print("借款原因 feature mean_importance: ",np.mean(Fea_impor0))
# print("年利率 feature mean_importance: ",np.mean(Fea_impor1))
# print("还款期限(月) feature mean_importance: ",np.mean(Fea_impor2))
# print("性别 feature mean_importance: ",np.mean(Fea_impor3))
# print("年龄 feature mean_importance: ",np.mean(Fea_impor4))
# print("婚姻 feature mean_importance: ",np.mean(Fea_impor5))
# print("收入 feature mean_importance: ",np.mean(Fea_impor6))
# print("工作时间 feature mean_importance: ",np.mean(Fea_impor7))
# print("学历 feature mean_importance: ",np.mean(Fea_impor8))
# print("房产 feature mean_importance: ",np.mean(Fea_impor9))
# print("房贷 feature mean_importance: ",np.mean(Fea_impor10))
# print("车产 feature mean_importance: ",np.mean(Fea_impor11))
# print("申请借款 feature mean_importance: ",np.mean(Fea_impor12))
# print("成功借款 feature mean_importance: ",np.mean(Fea_impor13))
# print("还清笔数 feature mean_importance: ",np.mean(Fea_impor14))
# print("信用额度 feature mean_importance: ",np.mean(Fea_impor15))
# print("借款总额 feature mean_importance: ",np.mean(Fea_impor16))
# sum=np.mean(Fea_impor0)+np.mean(Fea_impor1)+np.mean(Fea_impor2)+np.mean(Fea_impor3)+np.mean(Fea_impor4)+np.mean(Fea_impor5)+np.mean(Fea_impor6)+np.mean(Fea_impor7)+np.mean(Fea_impor8)+np.mean(Fea_impor9)+np.mean(Fea_impor10)+np.mean(Fea_impor11)+np.mean(Fea_impor12)+np.mean(Fea_impor13)+np.mean(Fea_impor14)+np.mean(Fea_impor15)+np.mean(Fea_impor16)
# print(sum)
print("finished")
# print("Feature importances:",clf.feature_importances_)
