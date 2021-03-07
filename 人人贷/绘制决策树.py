__author__ = 'YJY-1997'
# !/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc
from sklearn import tree
import graphviz
np.set_printoptions( suppress=True,linewidth=1000)#

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

dataMat = np.array(pd.read_excel('D://实验样本集.xls'))
y=dataMat[:,0]
X =dataMat[:,range(1,18,1)]
skf=StratifiedKFold(n_splits=10,random_state=1,shuffle=True)
clf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=20,min_impurity_decrease=0.005)#使用特征选择标准为entropy,默认为基尼系数”gini”,max_depth=8,
cnt = 0

for i, (train, test) in enumerate(skf.split(X,y)):       #利用模型划分数据集和目标变量 为一一对应的下标
    cnt +=1
    X_train,X_test=X[train,],X[test,]
    y_train,y_test=y[train,],y[test,]
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
    clf1=clf.fit(X_train_std, y_train)
    y_predict = clf.predict(X_test_std)
    print("Feature importances:",clf.feature_importances_)
    print(classification_report(y_predict,y_test,target_names=['CLOSED','BAD_DEBT']))
    feature_name = ['borrow_type','interest','months','gender','age','marriage','salary','work_years','graduation','has_house','house_loan','has_car','total_count','success_count','already_pay_count','available_credits','borrow_amount']

    # dot_data = tree.export_graphviz(clf, out_file=None,
    #                      feature_names=feature_name,
    #                      class_names=["CLOSED","BAD_DEBT",],
    #                      filled=True, rounded=True,
    #                      special_characters=True)
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # Image(graph.create_png())

    # with open("D://DT_gini.dot", 'w') as dot_data  :
    dot_data = tree.export_graphviz(clf
                               # ,out_file = dot_data
                               ,feature_names= feature_name
                               ,class_names=["CLOSED","BAD_DEBT",]
                               ,filled=True#让树的每一块有颜色，颜色越浅，表示不纯度越高
                               ,rounded=True#树的块的形状
                               ,special_characters=True)
    graph = graphviz.Source(dot_data)
    # import os
    # print(os.environ['PATH'])
    graph.render("Tree{}".format(cnt))
    graph.view()
    Fea_impor0.append(clf.feature_importances_[0])
    Fea_impor1.append(clf.feature_importances_[1])
    Fea_impor2.append(clf.feature_importances_[2])
    Fea_impor3.append(clf.feature_importances_[3])
    Fea_impor4.append(clf.feature_importances_[4])
    Fea_impor5.append(clf.feature_importances_[5])
    Fea_impor6.append(clf.feature_importances_[6])
    Fea_impor7.append(clf.feature_importances_[7])
    Fea_impor8.append(clf.feature_importances_[8])
    Fea_impor9.append(clf.feature_importances_[9])
    Fea_impor10.append(clf.feature_importances_[10])
    Fea_impor11.append(clf.feature_importances_[11])
    Fea_impor12.append(clf.feature_importances_[12])
    Fea_impor13.append(clf.feature_importances_[13])
    Fea_impor14.append(clf.feature_importances_[14])
    Fea_impor15.append(clf.feature_importances_[15])
    Fea_impor16.append(clf.feature_importances_[16])
    print(clf.feature_importances_)
    # if cnt==1:
    #     break

print("借款原因 feature mean_importance: ",np.mean(Fea_impor0))
print("年利率 feature mean_importance: ",np.mean(Fea_impor1))
print("还款期限(月) feature mean_importance: ",np.mean(Fea_impor2))
print("性别 feature mean_importance: ",np.mean(Fea_impor3))
print("年龄 feature mean_importance: ",np.mean(Fea_impor4))
print("婚姻 feature mean_importance: ",np.mean(Fea_impor5))
print("收入 feature mean_importance: ",np.mean(Fea_impor6))
print("工作时间 feature mean_importance: ",np.mean(Fea_impor7))
print("学历 feature mean_importance: ",np.mean(Fea_impor8))
print("房产 feature mean_importance: ",np.mean(Fea_impor9))
print("房贷 feature mean_importance: ",np.mean(Fea_impor10))
print("车产 feature mean_importance: ",np.mean(Fea_impor11))
print("申请借款 feature mean_importance: ",np.mean(Fea_impor12))
print("成功借款 feature mean_importance: ",np.mean(Fea_impor13))
print("还清笔数 feature mean_importance: ",np.mean(Fea_impor14))
print("信用额度 feature mean_importance: ",np.mean(Fea_impor15))
print("借款总额 feature mean_importance: ",np.mean(Fea_impor16))