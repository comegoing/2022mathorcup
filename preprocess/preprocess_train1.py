# coding=utf-8
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
# 填充空缺值
def my_fill(data):
    # print(data.isnull().sum()) #查看空值

    # 测试集中没有相关属性
    data.drop(['重定向次数', '重定向驻留时长','语音方式','是否去过营业厅','ARPU（家庭宽带）','是否实名登记用户'],axis=1,inplace=True)

    # 删除无关属性
    data.drop(['用户描述', '用户id', '用户描述.1'], axis=1,inplace=True)

    # 按照表格要求填充空缺值
    data['是否关怀用户'].fillna('否',inplace=True)

    # 该属性下的样本缺失值较多，且样本量较少直接删除（其它属性也缺失）
    drop1 = data[data["是否4G网络客户（本地剔除物联网）"].isnull()].index.tolist()
    data.drop(labels=drop1,axis=0,inplace=True)

    # 该属性下的样本格式错误，无法读取，导致变为0，由于是连续值，用均值代替
    drop2 = data[data["外省流量占比"].isnull()].index.tolist()
    temp_data = data.drop(labels=drop2,axis=0,inplace=False)
    fill = temp_data["外省流量占比"].unique()
    data["外省流量占比"].fillna(np.mean(fill),inplace=True)

    # 替换异常值0变为其他，这些特征其他属性没问题，在这里全是string类型的值里是0，可能是异常
    data['终端品牌'].replace(0, '其他', inplace=True)

    # 将两个投诉变为合并为一个投诉
    cnt1 = data['家宽投诉']+data['资费投诉']
    data['家宽投诉'] = cnt1
    data.drop(['资费投诉'],axis=1,inplace=True)
    data.rename(columns={'家宽投诉':'是否投诉'},inplace=True)
    data['是否投诉'] = data['是否投诉'].apply(lambda x: '否' if x == 0 else '是')

    # 将两个欠费变为合并为一个欠费
    cnt2 = data['当月欠费金额'] + data['前第3个月欠费金额']
    data['当月欠费金额'] = cnt2
    data.drop(['前第3个月欠费金额'], axis=1, inplace=True)
    data.rename(columns={'当月欠费金额': '是否不限量套餐到达用户'}, inplace=True)
    data['是否不限量套餐到达用户'] = data['是否不限量套餐到达用户'].apply(lambda x: '否' if x == 0 else '是')

    #重整行索引，为后面编码提供方便
    data.reset_index(drop=True,inplace=True)
    return data

# 属性编码
def my_encode(data):
    # 映射（层次）编码，对字符串属性进行特征编码
    code4 = {
        '2G':0,
        '4G':1,
        '5G':2
    }
    data['4\\5G用户'] = data['4\\5G用户'].map(code4)

    # 映射编码
    code6 = {
        '否':0,
        '是':1
    }
    data['是否关怀用户'] = data['是否关怀用户'].map(code6)
    data['是否4G网络客户（本地剔除物联网）'] = data['是否4G网络客户（本地剔除物联网）'].map(code6)
    data['是否5G网络客户'] = data['是否5G网络客户'].map(code6)
    data['是否投诉'] = data['是否投诉'].map(code6)
    data['是否不限量套餐到达用户'] = data['是否不限量套餐到达用户'].map(code6)

    # 映射编码
    val = data['终端品牌'].unique()
    labels = [i for i in range(0,len(val))]
    code14 = dict(zip(val,labels))
    data['终端品牌'] = data['终端品牌'].map(code14)

    #映射编码
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

data = pd.read_excel('../附件1语音业务用户满意度数据.xlsx')
data_fill = my_fill(data) #填充后的数据
data_encode = my_encode(data_fill) #编码后的数据

y1 = data_encode['语音通话整体满意度'] #总标签1
y2 = data_encode['网络覆盖与信号强度'] #标签2
y3 = data_encode['语音通话清晰度'] #标签3
y4 = data_encode['语音通话稳定性'] #标签4
y_all = pd.concat([y1,y2],axis=1)
y_all = pd.concat([y_all,y3],axis=1)
y_all = pd.concat([y_all,y4],axis=1)

# 去除标签的特征
x_dataframe = data_encode.drop(['语音通话整体满意度', '网络覆盖与信号强度','语音通话清晰度','语音通话稳定性'], axis=1)
x = pd.DataFrame(scalelize(x_dataframe))
x.columns = ['是否遇到过网络问题', '居民小区', '办公室', '高校', '商业街', '地铁', '农村', '高铁', '其他，请注明', '手机没有信号', '有信号无法拨通', '通话过程中突然中断', '通话中有杂音、听不清、断断续续', '串线', '通话过程中一方听不见', '其他，请注明.1', '脱网次数', 'mos质差次数', '未接通掉话次数', '是否投诉', '4\\5G用户', '是否关怀用户', '套外流量（MB）', '是否4G网络客户（本地剔除物联网）', '套外流量费（元）', '外省语音占比', '语音通话-时长（分钟）', '省际漫游-时长（分钟）', '终端品牌', '终端品牌类型', '当月ARPU', '当月MOU', '前3月ARPU', '前3月MOU', '外省流量占比', 'GPRS总流量（KB）', 'GPRS-国内漫游-流量（KB）', '是否5G网络客户', '客户星级标识', '是否不限量套餐到达用户']

# 导出
writer = pd.ExcelWriter("./1_train.xlsx")
x.to_excel(writer,index=False)
writer.save()

writer1 = pd.ExcelWriter("./1_train_label.xlsx")
y_all.to_excel(writer1,index=False)
writer1.save()



