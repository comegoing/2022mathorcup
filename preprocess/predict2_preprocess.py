# coding=utf-8
import pandas as pd
from sklearn.preprocessing import StandardScaler


# 填充空缺值
def my_fill(data):

    # 删除无关属性
    data.drop(['用户id','注明内容','注明内容.1','注明内容.2','注明内容.3','注明内容.4','学习强国','学习强国.1',
               '是否投诉','4\\5G用户','是否关怀用户','是否4G网络客户（本地剔除物联网）',
               '外省语音占比','语音通话-时长（分钟）','省际漫游-时长（分钟）',
               '前3月MOU','外省流量占比','GPRS总流量（KB）','GPRS-国内漫游-流量（KB）',
               '是否遇到网络问题'], axis=1,inplace=True)

    data.drop([data.columns.tolist()[19]],axis=1,inplace=True)
    # 填充为0
    data['上网质差次数'].fillna(0, inplace=True)
    data['脱网次数'].fillna(0, inplace=True)
    data['微信质差次数'].fillna(0, inplace=True)

    # 终端品牌类型的空缺值全是2G用户，所以赋予新值其他
    data['终端品牌类型'].fillna('其他', inplace=True)

    # 终端品牌没有其他选项，但是全是2G用户，所以赋予新值其他
    data['终端品牌'].fillna('其他', inplace=True)

    #重整行索引，为后面编码提供方便
    data.reset_index(drop=True,inplace=True)

    new_index = ['居民小区', '办公室', '高校', '商业街', '地铁', '农村', '高铁', '其他，请注明', '网络信号差/没有信号', '显示有信号上不了网', '上网过程中网络时断时续或时快时慢', '手机上网速度慢', '其他，请注明.1', '看视频卡顿', '打游戏延时大', '打开网页或APP图片慢', '下载速度慢', '手机支付较慢', '其他，请注明.2', '爱奇艺', '优酷', '腾讯视频', '芒果TV', '搜狐视频', '抖音', '快手', '火山', '咪咕视频', '其他，请注明.3', '全部都卡顿', '和平精英', '王者荣耀', '穿越火线', '梦幻西游', '龙之谷', '梦幻诛仙', '欢乐斗地主', '部落冲突', '炉石传说', '阴阳师', '其他，请注明.4', '全部游戏都卡顿', '微信', '手机QQ', '淘宝', '京东', '百度', '今日头条', '新浪微博', '拼多多', '其他，请注明.5', '全部网页或APP都慢', '上网质差次数', '脱网次数', '微信质差次数', '套外流量（MB）', '套外流量费（元）', '是否5G网络客户', '性别', '终端品牌', '终端品牌类型', '前3月ARPU', '当月ARPU', '当月MOU', '客户星级标识', '是否不限量套餐到达用户']
    new_data = data.reindex(columns=new_index)

    return new_data

# 属性编码
def my_encode(data):

    # 映射编码
    val1 = data['是否5G网络客户'].unique()
    labels1 = [i for i in range(0,len(val1))]
    code6 = dict(zip(val1,labels1))
    data['是否5G网络客户'] = data['是否5G网络客户'].map(code6)
    data['是否不限量套餐到达用户'] = data['是否不限量套餐到达用户'].map(code6)

    # 映射编码
    val2 = data['性别'].unique()
    labels2 = [i for i in range(0, len(val2))]
    code7 = dict(zip(val2, labels2))
    data['性别'] = data['性别'].map(code7)

    # 映射编码
    val2 = data['终端品牌'].unique()
    labels2 = [i for i in range(0, len(val2))]
    code7 = dict(zip(val2, labels2))
    data['终端品牌'] = data['终端品牌'].map(code7)

    # 映射编码
    val2 = data['终端品牌类型'].unique()
    labels2 = [i for i in range(0, len(val2))]
    code7 = dict(zip(val2, labels2))
    data['终端品牌类型'] = data['终端品牌类型'].map(code7)

    # 映射编码
    val2 = data['客户星级标识'].unique()
    labels2 = [i for i in range(0, len(val2))]
    code7 = dict(zip(val2, labels2))
    data['客户星级标识'] = data['客户星级标识'].map(code7)

    return data
def scalelize(data):
    scal = StandardScaler()
    new_data = scal.fit_transform(data)
    return new_data

data = pd.read_excel('../附件4上网业务用户满意度预测数据.xlsx')
data_fill = my_fill(data)
data_encode = my_encode(data_fill) #填充后的数据
predict2 = pd.DataFrame(scalelize(data_encode))
predict2.columns = ['居民小区', '办公室', '高校', '商业街', '地铁', '农村', '高铁', '其他，请注明', '网络信号差/没有信号', '显示有信号上不了网', '上网过程中网络时断时续或时快时慢', '手机上网速度慢', '其他，请注明.1', '看视频卡顿', '打游戏延时大', '打开网页或APP图片慢', '下载速度慢', '手机支付较慢', '其他，请注明.2', '爱奇艺', '优酷', '腾讯视频', '芒果TV', '搜狐视频', '抖音', '快手', '火山', '咪咕视频', '其他，请注明.3', '全部都卡顿', '和平精英', '王者荣耀', '穿越火线', '梦幻西游', '龙之谷', '梦幻诛仙', '欢乐斗地主', '部落冲突', '炉石传说', '阴阳师', '其他，请注明.4', '全部游戏都卡顿', '微信', '手机QQ', '淘宝', '京东', '百度', '今日头条', '新浪微博', '拼多多', '其他，请注明.5', '全部网页或APP都慢', '上网质差次数', '脱网次数', '微信质差次数', '套外流量（MB）', '套外流量费（元）', '是否5G网络客户', '性别', '终端品牌', '终端品牌类型', '前3月ARPU', '当月ARPU', '当月MOU', '客户星级标识', '是否不限量套餐到达用户']

# 导出
writer = pd.ExcelWriter("./2_predict_process.xlsx")
predict2.to_excel(writer,index=False)
writer.save()








