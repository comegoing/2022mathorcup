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
def model_train(x,y):
    model = DecisionTreeClassifier(criterion='gini')
    model.fit(x, y)
    property_weight = model.feature_importances_.tolist()
    w = pd.concat([pd.DataFrame(property_weight,columns=['值域']),pd.DataFrame(x.columns.tolist(),columns=['属性'])],axis=1)
    w = w.sort_values('值域',ascending=False)
    plt.bar(w['属性'].tolist(), w['值域'].tolist(), width=0.8)
    plt.title(y.columns.tolist()[0], FontProperties=font)
    plt.xticks(rotation=90, FontProperties=font,fontsize=6)
    plt.show()
    return 0

x = pd.read_excel('../preprocess/1_train.xlsx')
y = pd.read_excel('../preprocess/1_train_label.xlsx')
y1 = y.drop(['网络覆盖与信号强度','语音通话清晰度','语音通话稳定性'],axis=1)
y2 = y.drop(['语音通话整体满意度','语音通话清晰度','语音通话稳定性'],axis=1)
y3 = y.drop(['语音通话整体满意度','网络覆盖与信号强度','语音通话稳定性'],axis=1)
y4 = y.drop(['语音通话整体满意度','网络覆盖与信号强度','语音通话清晰度'],axis=1)
# 这里是复制的第二个关联度的数据，要修改
# 评分1
model_train(x,y1)
# 评分2
model_train(x,y2)
# 评分3
model_train(x,y3)
# 评分4
model_train(x,y4)




