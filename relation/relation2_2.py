import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

font = FontProperties(fname="C:\\windows\\fonts\\simsun.ttc")
def model_train(x,y):
    model = DecisionTreeClassifier(criterion='gini')
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

x = pd.read_excel('../preprocess/2_train.xlsx')
y = pd.read_excel('../preprocess/2_train_label.xlsx')
y1 = y.drop(['网络覆盖与信号强度','手机上网速度','手机上网稳定性'],axis=1)
y2 = y.drop(['手机上网整体满意度','手机上网速度','手机上网稳定性'],axis=1)
y3 = y.drop(['手机上网整体满意度','网络覆盖与信号强度','手机上网稳定性'],axis=1)
y4 = y.drop(['手机上网整体满意度','网络覆盖与信号强度','手机上网速度'],axis=1)
# 评分1
model_train(x,y1)
# 评分2
model_train(x,y2)
# 评分3
model_train(x,y3)
# 评分4
model_train(x,y4)





