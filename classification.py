import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')
test_label = pd.read_csv('./data/gender_submission.csv')
result = open('./result/classification_result.txt','a+')

#数据预处理
attr= ['Pclass','Sex','Age','SibSp','Parch','Embarked']
train = train_data[attr]
train_label = train_data['Survived']
test = test_data[attr]
test_r = test_label['Survived']

train['Age'].fillna(train['Age'].mean(),inplace=True)
train['Embarked'].fillna(train['Embarked'].value_counts().index[0],inplace = True)
test['Age'].fillna(test['Age'].mean(),inplace=True)
test['Embarked'].fillna(test['Embarked'].value_counts().index[0],inplace = True)

#对类别型特征进行转化,成为特征向量
vec=DictVectorizer(sparse=False)
_train=vec.fit_transform(train.to_dict(orient='record'))
_test=vec.fit_transform(test.to_dict(orient='record'))

#使用单一决策树进行模型训练以及预测分析
param_dtc = {'min_samples_leaf':range(5,31,5)}
dtc=DecisionTreeClassifier(criterion='gini',splitter='best')
grid_dtc = GridSearchCV(estimator=dtc,param_grid=param_dtc,cv=5)
grid_dtc.fit(_train,train_label)
dtc_pred=grid_dtc.predict(_test)
print('DecisionTreeClassifier:\n',classification_report(dtc_pred,test_r))
print('DecisionTreeClassifier:\n',classification_report(dtc_pred,test_r),file=result)

#使用随机森林分类器进行集成模型的训练以及预测分析
param_rfc = {'n_estimators':range(10,101,10)}
rfc=RandomForestClassifier()
grid_rfc = GridSearchCV(estimator=rfc,param_grid=param_rfc,cv=5)
grid_rfc.fit(_train,train_label)
rfc_pred=grid_rfc.predict(_test)
print('RandomForestClassifier:\n',classification_report(rfc_pred,test_r))
print('RandomForestClassifier:\n',classification_report(rfc_pred,test_r),file=result)

#使用梯度提升决策树进行集成模型的训练以及预测分析
param_gbc = {'n_estimators':range(10,101,10),'max_depth':range(3,14,2)}
gbc=GradientBoostingClassifier()
grid_gbc = GridSearchCV(estimator=gbc,param_grid=param_gbc,cv=5)
grid_gbc.fit(_train,train_label)
gbc_pred=grid_gbc.predict(_test)
print('GradientBoostingClassifier:\n',classification_report(gbc_pred,test_r))
print('GradientBoostingClassifier:\n',classification_report(gbc_pred,test_r),file=result)


color_value = lambda a: 'r' if a == 1 else 'b'
#3D绘图
#单一决策树绘图
color_dtc= [color_value(d) for d in dtc_pred]
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(test['Parch'],test['SibSp'],test['Pclass'],c = color_dtc,marker = 'o')
ax.set_xlabel('parch')
ax.set_ylabel('sibsp')
ax.set_zlabel('pclass')
plt.savefig('./plot/classification_result1_dtc.png')
plt.clf()

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(test['Age'],test['SibSp'],test['Pclass'],c = color_dtc,marker = 'o')
ax.set_xlabel('age')
ax.set_ylabel('sibsp')
ax.set_zlabel('pclass')
plt.savefig('./plot/classification_result2_dtc.png')
plt.clf()


#随机森林分类器绘图
color_rfc= [color_value(d) for d in rfc_pred]
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(test['Parch'],test['SibSp'],test['Pclass'],c = color_rfc,marker = 'o')
ax.set_xlabel('parch')
ax.set_ylabel('sibsp')
ax.set_zlabel('pclass')
plt.savefig('./plot/classification_result1_rfc.png')
plt.clf()

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(test['Age'],test['SibSp'],test['Pclass'],c = color_rfc,marker = 'o')
ax.set_xlabel('age')
ax.set_ylabel('SibSp')
ax.set_zlabel('pclass')
plt.savefig('./plot/classification_result2_rfc.png')
plt.clf()

#梯度提升决策树绘图
color_gbc= [color_value(d) for d in gbc_pred]
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(test['Parch'],test['SibSp'],test['Pclass'],c = color_gbc,marker = 'o')
ax.set_xlabel('parch')
ax.set_ylabel('sibsp')
ax.set_zlabel('pclass')
plt.savefig('./plot/classification_result1_gbc.png')
plt.clf()

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(test['Age'],test['SibSp'],test['Pclass'],c = color_gbc,marker = 'o')
ax.set_xlabel('age')
ax.set_ylabel('SibSp')
ax.set_zlabel('pclass')
plt.savefig('./plot/classification_result2_gbc.png')
plt.clf()

