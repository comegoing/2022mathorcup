import pandas as pd
from sklearn.model_selection import train_test_split,KFold,GridSearchCV
from sklearn.metrics import f1_score,make_scorer,accuracy_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
font = FontProperties(fname="C:\\windows\\fonts\\simsun.ttc")
# 模型训练，绘图出属性权重，打印准确率
def model_train(x,y,a,b,c):
    model = DecisionTreeClassifier(criterion='gini',max_depth=a,min_samples_leaf=b,min_samples_split=c)
    model.fit(x, y)
    property_weight = model.feature_importances_.tolist()
    w = pd.concat([pd.DataFrame(property_weight, columns=['值域']), pd.DataFrame(x.columns.tolist(), columns=['属性'])],
                  axis=1)
    w = w.sort_values('值域', ascending=False)
    plt.bar(w['属性'].tolist(), w['值域'].tolist(), width=0.8)
    plt.title(y.columns.tolist()[0], FontProperties=font)
    plt.xticks(rotation=90, FontProperties=font, fontsize=6)
    plt.show()
    return 0

# 下面几个函数都是参数训练
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
x = pd.read_excel('../preprocess/1_train.xlsx')
y = pd.read_excel('../preprocess/1_train_label.xlsx')
y1 = y.drop(['网络覆盖与信号强度','语音通话清晰度','语音通话稳定性'],axis=1)
y2 = y.drop(['语音通话整体满意度','语音通话清晰度','语音通话稳定性'],axis=1)
y3 = y.drop(['语音通话整体满意度','网络覆盖与信号强度','语音通话稳定性'],axis=1)
y4 = y.drop(['语音通话整体满意度','网络覆盖与信号强度','语音通话清晰度'],axis=1)
# 这里是复制的第二个关联度的数据，要修改
# 评分1
model_train(x,y1,5,106,787)
# 评分2
model_train(x,y2,5,155,436)
# 评分3
model_train(x,y3,5,257,572)
# 评分4
model_train(x,y4,6,288,610)

# 三个函数分别得到三个参数，然后把值填倒上面的评分中
# a = max_depth(x,y3)
# b = min_samples_leaf(x,y3,a)
# c = min_samples_split(x,y3,a,b)



