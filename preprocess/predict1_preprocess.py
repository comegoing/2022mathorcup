import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 填充空缺值
def my_fill(data):

    # 删除无关属性
    data.drop(['用户描述', '用户id', '用户描述.1','性别'], axis=1,inplace=True)

    #重整行索引，为后面编码提供方便
    data.reset_index(drop=True,inplace=True)
    return data

# 属性编码
def my_encode(data):
    # 映射（层次）编码，对字符串属性进行特征编码
    code4 = {
        '2G': 0,
        '4G': 1,
        '5G': 2
    }
    data['4\\5G用户'] = data['4\\5G用户'].map(code4)

    # 映射编码
    code6 = {
        '否': 0,
        '是': 1
    }
    data['是否关怀用户'] = data['是否关怀用户'].map(code6)
    data['是否4G网络客户（本地剔除物联网）'] = data['是否4G网络客户（本地剔除物联网）'].map(code6)
    data['是否5G网络客户'] = data['是否5G网络客户'].map(code6)
    data['是否投诉'] = data['是否投诉'].map(code6)
    data['是否不限量套餐到达用户'] = data['是否不限量套餐到达用户'].map(code6)

    # 映射编码
    val = data['终端品牌'].unique()
    labels = [i for i in range(0, len(val))]
    code14 = dict(zip(val, labels))
    data['终端品牌'] = data['终端品牌'].map(code14)

    # 映射编码
    val = data['终端品牌类型'].unique()
    labels = [i for i in range(0, len(val))]
    code15 = dict(zip(val, labels))
    data['终端品牌类型'] = data['终端品牌类型'].map(code15)

    # 映射编码
    val = data['客户星级标识'].unique()
    labels = [i for i in range(0, len(val))]
    code18 = dict(zip(val, labels))
    data['客户星级标识'] = data['客户星级标识'].map(code18)

    return data
def scalelize(data):
    scal = StandardScaler()
    new_data = scal.fit_transform(data)
    return new_data

data = pd.read_excel('../附件3语音业务用户满意度预测数据.xlsx')
data_fill = my_fill(data) #填充后的数据
data_encode = my_encode(data_fill) #编码后的数据
predict1 = pd.DataFrame(scalelize(data_encode))
predict1.columns = data_encode.columns.tolist()
# 导出
writer = pd.ExcelWriter("./1_predict_process.xlsx")
predict1.to_excel(writer,index=False)
writer.save()




