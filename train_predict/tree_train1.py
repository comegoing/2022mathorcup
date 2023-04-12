import pandas as pd
from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from sklearn.metrics import f1_score,make_scorer,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties(fname="C:\\windows\\fonts\\simsun.ttc")
def model_train(x,y,a,b,c,x_pre):
    model = DecisionTreeClassifier(criterion='gini', max_depth=a, min_samples_leaf=b, min_samples_split=c)
    model.fit(x, y)
    y_pre = model.predict(x_pre)
    return y_pre

def maxdepth(x,y):
    model = DecisionTreeClassifier(criterion='gini')
    para = {'max_depth':range(1,50)}
    KF = KFold(10)
    grid = GridSearchCV(model,para,cv=KF)
    grid = grid.fit(x,y)
    reg = grid.best_estimator_
    for key in para.keys():
        best = reg.get_params()[key]
        print('最佳参数{0}:{1}'.format(key,reg.get_params()[key]))
    return best
def min_samples_leaf(x,y,i):
    model = DecisionTreeClassifier(criterion='gini',max_depth=i)
    para = {'min_samples_leaf':range(1,500)}
    KF = KFold(10)
    grid = GridSearchCV(model,para,cv=KF)
    grid = grid.fit(x,y)
    reg = grid.best_estimator_
    for key in para.keys():
        best = reg.get_params()[key]
        print('最佳参数{0}:{1}'.format(key,reg.get_params()[key]))
    return best
def min_samples_split(x,y,i,j):
    model = DecisionTreeClassifier(criterion='gini',max_depth=i,min_samples_leaf=j)
    para = {'min_samples_split':range(2,900)}
    KF = KFold(10)
    grid = GridSearchCV(model,para,cv=KF)
    grid = grid.fit(x,y)
    reg = grid.best_estimator_
    for key in para.keys():
        best = reg.get_params()[key]
        print('最佳参数{0}:{1}'.format(key,reg.get_params()[key]))
    return best
def load_out(x,y,a,b,c,x_pre):
    y_pre = model_train(x, y, a, b, c, x_pre)
    y_dataframe = pd.DataFrame(y_pre)
    y_dataframe.columns = y.columns.tolist()

    return y_dataframe
x = pd.read_excel('../preprocess/1_train.xlsx')
y = pd.read_excel('../preprocess/1_train_label.xlsx')
y1 = y.drop(['网络覆盖与信号强度','语音通话清晰度','语音通话稳定性'],axis=1)
y2 = y.drop(['语音通话整体满意度','语音通话清晰度','语音通话稳定性'],axis=1)
y3 = y.drop(['语音通话整体满意度','网络覆盖与信号强度','语音通话稳定性'],axis=1)
y4 = y.drop(['语音通话整体满意度','网络覆盖与信号强度','语音通话清晰度'],axis=1)
# 导出1
x_pre = pd.read_excel('../preprocess/1_predict_process.xlsx')
pre1 = load_out(x,y1,4,227,2,x_pre)
pre2 = load_out(x,y2,1,1,2,x_pre)
pre3 = load_out(x,y3,1,1,2,x_pre)
pre4 = load_out(x,y4,4,237,497,x_pre)
predict = pd.concat([pre1,pre2],axis=1)
predict = pd.concat([predict,pre3],axis=1)
predict = pd.concat([predict,pre4],axis=1)
writer = pd.ExcelWriter('../result1.xlsx')
predict.to_excel(writer,index=False)
writer.save()








