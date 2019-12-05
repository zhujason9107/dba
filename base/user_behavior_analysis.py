#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# + "user_id":用户ID
# + "item_id":商品ID
# + "behavior_type":用户地址
# + "item_category":商品类别
# + "time":登录时间
#
# + behavior_type: 1点击 2收藏 3购物车 4购买
# + geohash: 地理位置编码理解为经纬度的字符串模式

# ##### 流量指标：PV(APP总访问量)  、 UV (APP独立访问数)
# #### 流量指标还可以根据时间、日期等进行追踪、查看用户访问规律，即用户访问APP有何规律

file_path="tianchi_mobile_recommend_train_user 1.csv"
names=["user_id","item_id","behavior_type","user_geohash","item_category","time"]
taobaodata=pd.read_csv("tianchi_mobile_recommend_train_user 1.csv",encoding="utf8",header=None,names=names)

taobaodata.head()

taobaodata.info()

taobaodata.shape  #(12312542,6)
taobaodata.isnull().sum()["user_geohash"]  # 8402567  占总样本数的约68%
#缺失值处理
taobaodata.drop(columns="user_geohash",inplace=True)    # taobaodata.drop("user_geohash",axis=1,inplace=True) ,axis=指定是删除user_geohash所在的行还是所在的列

#数据类型转换
taobaodata["user_id"]=taobaodata["user_id"].astype(int)
taobaodata["item_id"]=taobaodata["item_id"].astype(int)
taobaodata["behavior_type"]=taobaodata["behavior_type"].astype(int)
taobaodata["item_category"]=taobaodata["item_category"].astype(int)

taobaodata.describe()

taobaodata.info()

# 异常值检测 画散点图 数据量太大不建议
# taobaodata.plot()
# 第二种方式检测异常值，根据业务知识，比如userid，item_id ，behavior_type，item_category 不能小于0

# taobaodata.query("item_category<0")  # 得知 暂未发现小于0的值

taobaodata["behavior_type"].value_counts()

#删除重复行
print(taobaodata.count())
taobaodata.drop_duplicates(["user_id","item_id","behavior_type","item_category","time"],keep="first") # keep="first" 遇到重复留下第一个
taobaodata.count()

#关于behavior_type的处理
# {1:"点击",2:"收藏",3:"购物车",4:"购买"}
t={1:"pv",2:"fav",3:"cart",4:"buy"}
taobaodata["behavior_type"]=taobaodata["behavior_type"].map(lambda x:t.get(x,x))

taobaodata.head()

# 时间处理 因为time列的类型是Object
taobaodata["date"]=taobaodata["time"].map(lambda x:x.split(" ")[0])
taobaodata["hour"]=taobaodata["time"].map(lambda x:x.split(" ")[1])

taobaodata.head()

# 总 pv
taobaodata[taobaodata["behavior_type"]=="pv"].count()["user_id"]
# taobaodata.groupby("behavior_type").count()["user_id"]

# 总 uv
# taobaodata["user_id"].unique
taobaodata.groupby("user_id").sum().count()["item_id"]

#  时间段的pv uv
df_pv=taobaodata[taobaodata["behavior_type"]=="pv"] #把只含pv的数据当成新的dataframe
df_pv.groupby("date").count()["user_id"]

df_pv.groupby(["date","user_id"]).count()


# + behavior_type: 1点击 2收藏 3购物车 4购买
# +          {1:"pv",2:"fav",3:"cart",4:"buy"}
# + 分析各个流程的转化漏斗，转化率为多少  （漏斗模型）
# + 购买的用户数及用户的复购率为多少  （复购率）
# + 应该对哪些用户做运营服务  （RFM分层模型）

stat=taobaodata.groupby("behavior_type").count()["user_id"]
pv=stat["pv"]
fav=stat["fav"]+stat["cart"]
buy=stat["buy"]

print(pv,fav,buy)

r1=fav*100/pv
r2=buy*100/fav

taobaodata.groupby("userid").min()["time"]

taobaodata["date"]=pd.to_datetime(taobaodata["date"]) #format="%Y%m%d"
#pd.datetime可以将特定的字符串或者数字转换成时间格式，其中format参数用于匹配
#例如19970201，%Y匹配四位数字1997，如果小写y只匹配两位数字97，%m匹配02，%d匹配01
#另外，小时是%h,分钟是%M,注意和月的大小写不一致，秒是%s.若是1997-02-01这样的形式，则是%Y-%m-%d。以此类推

taobaodata.info()
taobaodata["month"]=taobaodata["date"].values.astype("datetime64[M]")
#datetime64[M] （月精度）
#datetime64[ns] （纳秒精度）
#datetime64[D] （日精度）
#astype也可以将时间格式进行转换，比如[M]转化成月份。我们将月份作为消费行为的主要事件窗口，选择哪种时间窗口取决于消费频率
taobaodata.head()

# 用户首次购买
taobaodata.groupby("user_id").min()["date"]

# 所有用户首次购买集中在哪个月份
taobaodata.groupby("user_id").min()["month"].value_counts()

# 用户最后一次购买
taobaodata.groupby("user_id").max()["date"]

# 所有用最后一次购买集中在哪个月份
taobaodata.groupby("user_id").max()["month"].value_counts()

# ### 分析消费中的复购率和回购率。首先将用户消费数据进行数据透视

# 通过数据透视表，每个用户每个月消费次数
pivoted_counts=taobaodata.pivot_table(index="user_id",columns="month",values="date",aggfunc="count").fillna(0)
# values="date"  对date 当成值进行计数的原因，因为是基于用户id分组，统计date出现的次数 即统计每个用户有多少订单
# 整个透视表的大意，每个用户 每个月的订单数
# fillna(0) 指定0 填充缺失值 ，有些用户不是每个月都有消费记录，所以没有消费的月份里，填充0

# 因为pivoted_counts列的月份是2014-11-01 00:00:00 表示太丑，所以优化列名称展示
columns_month=taobaodata["month"].sort_values().map(lambda x:x.year*100+x.month).unique()
# unique()  别漏括号

pivoted_counts.columns=columns_month

pivoted_counts


# 求复购率： 单位时间内消费两次及以上的用户 在总消费用户中占比
# 将数据转换一下，消费两次及以上记为1，消费一次记为0，没有消费记为NaN
pivoted_counts_transf=pivoted_counts.applymap(lambda x:1 if x>1 else np.NaN if x==0 else 0)
# pivoted_counts_transf

#计算复购率
(pivoted_counts_transf.sum()/pivoted_counts_transf.count()).plot(figsize=(10,4))

# (lambda x:1 if x>1 else np.NaN if x==0 else 0)
# def func(x):
#     if x>1:
#         return 1
#     elif x==0:
#         return np.NaN
#     else:
#         return 0
# 因为 lambda 没有elif 用法

# 回购率 的计算：
# 回购率是某一个时间窗口内消费的用户，在下一个时间窗口仍旧消费的占比。
# 我1月消费用户1000，他们中有300个2月依然消费，回购率是30%
pivoted_purchase=pivoted_counts.applymap(lambda x:1 if x>1 else 0)
pivoted_purchase

#提取 有购买行为的 用户
user_data=taobaodata[taobaodata["behavior_type"]=="buy"]
# 将有购买行为的用户 进行分组 计数 得到 每个有购买行为的用户 的所有购买次数
user_data=user_data.groupby("user_id").count()["item_id"]
total=user_data.count()

# 购买次数大于2 才叫复购，求出 将购买次数大于等于2的 值变为1，小于2的变为0，因为要求复购率：单位时间内 购买次数大于等于2 的用户数量 占 总购买用户数量的比
total2=user_data.map(lambda x:1 if x>=2 else 0).sum()
# 求复购率
fugou=total2/total
fugou


# + RFM
# + 简化
# + F值
# + 购买1-5次 f1 新客
# + 购买6次以上 f2 忠诚
#
# + R值
# + 7天活跃 r1 活跃
# + 7天以上 r2 流失
#
# + f1r1,f1r2,f2r1,f2r2 四层
#






