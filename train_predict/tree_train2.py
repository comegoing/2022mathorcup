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
x = pd.read_excel('../preprocess/2_train.xlsx')
y = pd.read_excel('../preprocess/2_train_label.xlsx')
y1 = y.drop(['网络覆盖与信号强度','手机上网速度','手机上网稳定性'],axis=1)
y2 = y.drop(['手机上网整体满意度','手机上网速度','手机上网稳定性'],axis=1)
y3 = y.drop(['手机上网整体满意度','网络覆盖与信号强度','手机上网稳定性'],axis=1)
y4 = y.drop(['手机上网整体满意度','网络覆盖与信号强度','手机上网速度'],axis=1)
# 导出1
x_pre = pd.read_excel('../preprocess/2_predict_process.xlsx')
pre1 = load_out(x,y1,5,106,787,x_pre)
pre2 = load_out(x,y2,5,155,436,x_pre)
pre3 = load_out(x,y3,5,257,572,x_pre)
pre4 = load_out(x,y4,6,288,610,x_pre)
# predict = pd.concat([pre1,pre2],axis=1)
# predict = pd.concat([predict,pre3],axis=1)
# predict = pd.concat([predict,pre4],axis=1)
# writer = pd.ExcelWriter('../result2.xlsx')
# predict.to_excel(writer,index=False)
# writer.save()
# # 评分2
# model_train(x,y2,5,155,436)
# # 评分3
# model_train(x,y3,5,257,572)
# # 评分4
# model_train(x,y4,6,288,610)

# a = max_depth(x,y3)
# b = min_samples_leaf(x,y3,a)
# c = min_samples_split(x,y3,a,b)





