# coding=utf-8
import pandas as pd
from sklearn.preprocessing import StandardScaler

# 填充空缺值
def my_fill(data):

    # 删除无关属性
    data.drop(['场景备注数据', '用户', '现象备注数据', 'APP大类备注',
               'APP小类视频备注','APP小类游戏备注','APP小类上网备注',
               '重定向次数','2G驻留时长','王者荣耀质差次数','高单价超套客户（集团）',
               '高频高额超套客户（集团）','是否全月漫游用户','年龄',
               '王者荣耀使用次数','游戏类APP使用天数','游戏类APP使用次数',
               '王者荣耀使用天数','游戏类APP使用流量','抖音使用流量（MB）',
               '今日头条使用流量','快手使用流量','优酷视频使用流量',
               '腾讯视频使用流量','小视频系APP流量','阿里系APP流量',
               '网易系APP流量','腾讯系APP流量','王者荣耀APP使用流量',
               '蜻蜓FMAPP使用流量','饿了么使用流量','美团外卖使用流量',
               '天猫使用流量','大众点评使用流量','滴滴出行使用流量',
               '通信类应用流量','游戏类应用流量','网页类应用流量',
               '音乐类应用流量','视频类应用流量','邮箱类应用流量',
               '终端类型','操作系统','终端制式','当月GPRS资源使用量（GB）',
               '是否校园套餐用户','校园卡无校园合约用户',
               '校园卡校园合约捆绑用户','当月高频通信分公司',
               '畅享套餐档位','畅享套餐名称','主套餐档位',
               '近3个月平均消费（剔除通信账户支付）','码号资源-激活时间',
               '码号资源-发卡时间'], axis=1,inplace=True)

    # 填充为0
    data['上网质差次数'].fillna(0, inplace=True)
    data['脱网次数'].fillna(0, inplace=True)
    data['微信质差次数'].fillna(0, inplace=True)

    # 终端品牌类型的空缺值全是2G用户，所以赋予新值其他
    data['终端品牌类型'].fillna('其他', inplace=True)

    # 终端品牌没有其他选项，但是全是2G用户，所以赋予新值其他
    data['终端品牌'].fillna('其他', inplace=True)

    # arpu就是消费
    data.rename(columns={'近3个月平均消费（元）':'前3月ARPU'},inplace=True)

    #本月消费可以由本年消费求平均
    data['本年累计消费（元）'] = data['本年累计消费（元）']/12
    data.rename(columns={'本年累计消费（元）':'当月ARPU'},inplace=True)

    #重整行索引，为后面编码提供方便
    data.reset_index(drop=True,inplace=True)

    # 重整列名，匹配预测集
    new_index = ['手机上网整体满意度', '网络覆盖与信号强度', '手机上网速度', '手机上网稳定性', '居民小区', '办公室', '高校', '商业街', '地铁', '农村', '高铁', '其他，请注明', '网络信号差/没有信号', '显示有信号上不了网', '上网过程中网络时断时续或时快时慢', '手机上网速度慢', '其他，请注明.1', '看视频卡顿', '打游戏延时大', '打开网页或APP图片慢', '下载速度慢', '手机支付较慢', '其他，请注明.2', '爱奇艺', '优酷', '腾讯视频', '芒果TV', '搜狐视频', '抖音', '快手', '火山', '咪咕视频', '其他，请注明.3', '全部都卡顿', '和平精英', '王者荣耀', '穿越火线', '梦幻西游', '龙之谷', '梦幻诛仙', '欢乐斗地主', '部落冲突', '炉石传说', '阴阳师', '其他，请注明.4', '全部游戏都卡顿', '微信', '手机QQ', '淘宝', '京东', '百度', '今日头条', '新浪微博', '拼多多', '其他，请注明.5', '全部网页或APP都慢', '上网质差次数', '脱网次数', '微信质差次数', '套外流量（MB）', '套外流量费（元）', '是否5G网络客户', '性别', '终端品牌', '终端品牌类型', '前3月ARPU', '当月ARPU', '当月MOU', '客户星级标识', '是否不限量套餐到达用户']
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

data = pd.read_excel('../附件2上网业务用户满意度数据.xlsx')
data_fill = my_fill(data)
data_encode = my_encode(data_fill) #填充后的数据
y1 = data_encode['手机上网整体满意度'] #总标签1
y2 = data_encode['网络覆盖与信号强度'] #标签2
y3 = data_encode['手机上网速度'] #标签3
y4 = data_encode['手机上网稳定性'] #标签4
y_all = pd.concat([y1,y2],axis=1)
y_all = pd.concat([y_all,y3],axis=1)
y_all = pd.concat([y_all,y4],axis=1)

# 去除标签的特征
x_dataframe = data_encode.drop(['手机上网整体满意度', '网络覆盖与信号强度','手机上网速度','手机上网稳定性'], axis=1)
x = pd.DataFrame(scalelize(x_dataframe))
x.columns = ['居民小区', '办公室', '高校', '商业街', '地铁', '农村', '高铁', '其他，请注明', '网络信号差/没有信号', '显示有信号上不了网', '上网过程中网络时断时续或时快时慢', '手机上网速度慢', '其他，请注明.1', '看视频卡顿', '打游戏延时大', '打开网页或APP图片慢', '下载速度慢', '手机支付较慢', '其他，请注明.2', '爱奇艺', '优酷', '腾讯视频', '芒果TV', '搜狐视频', '抖音', '快手', '火山', '咪咕视频', '其他，请注明.3', '全部都卡顿', '和平精英', '王者荣耀', '穿越火线', '梦幻西游', '龙之谷', '梦幻诛仙', '欢乐斗地主', '部落冲突', '炉石传说', '阴阳师', '其他，请注明.4', '全部游戏都卡顿', '微信', '手机QQ', '淘宝', '京东', '百度', '今日头条', '新浪微博', '拼多多', '其他，请注明.5', '全部网页或APP都慢', '上网质差次数', '脱网次数', '微信质差次数', '套外流量（MB）', '套外流量费（元）', '是否5G网络客户', '性别', '终端品牌', '终端品牌类型', '前3月ARPU', '当月ARPU', '当月MOU', '客户星级标识', '是否不限量套餐到达用户']

# 导出
writer = pd.ExcelWriter("./2_train.xlsx")
x.to_excel(writer,index=False)
writer.save()

writer1 = pd.ExcelWriter("./2_train_label.xlsx")
y_all.to_excel(writer1,index=False)
writer1.save()









