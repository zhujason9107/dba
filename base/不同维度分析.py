#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# ### 一、明确分析问题
# + 1、订单维度：笔单价和连带率是多少？订单金额与订单内商品件数的关系如何
# + 2、客户维度：客单价是多少？客单价消费金额与消费件数的关系如何
#
# + 相关分析1
# + x=订单商品数量
# + y=某个x订单总额平均
#
# + 相关分析2
# + x=客户总件数
# + y=某个x下订单总额平均

# ### 二、获取数据 、探索分析

df=pd.read_csv("data_utf8.csv")


df.head()

df.info()


df.shape


# 替换字段名称
df.columns

df.columns=list(["发票号","产品编码","产品描述","产品数量","发票日期","产品单价","用户ID","国家"])


df.head()


df.info()

#修改 发票日期 字段 数据 类型为 datetime
df["发票日期"]=pd.to_datetime(df["发票日期"])


df["用户ID"]=df["用户ID"].astype(object)

df.info()

df.describe()


# + 1、产品数量平均值为9.552250，存在极值（异常值）干扰，中位数是3比平均值小，说明数据右偏
# + 2、产品单价平均值为4.611114，存在极值干扰，中位数是2.08比平均值小，说明数据右偏

df["总价"]=df["产品数量"]*df["产品单价"]
df.head()

# 画散点图 查看异常值
plt.rcParams["font.sans-serif"]=["SimHei"]
plt.rcParams["axes.unicode_minus"]=False
plt.figure(figsize=(20,8),dpi=80)
plt.grid()
plt.scatter(df["产品数量"],df["产品单价"])
plt.xlabel("产品数量")
plt.ylabel("产品单价")


# + 根据散点图观察可知，存在异常值，为了避免异常值干扰可采用切比雪夫定律，产品数量范围可以选取(0,20000),产品单价范围可选取(0,20000)
# + 处理异常值可先查看异常值记录的数量，数量小则采取删除异常值所在记录，数量大则采取填充平均值 、众数等方式

# ### 三、数据预处理

#提取 产品数量 0到20000 的行
df_data=df.query(("产品数量 > 0 & 产品数量< 20000"))

df_data.info()

df.isnull().count()
# df.fillna(values=5) 填充固定值

#提取 产品单价 0到20000 的行
df_finshdata=df_data.query(("产品单价 > 0 & 产品单价< 20000"))



df_finshdata.info()


df_finshdata

plt.figure(figsize=(20,8),dpi=80)
plt.grid()
plt.scatter(df_finshdata["产品数量"],df_finshdata["产品单价"])
plt.xlabel("产品数量")
plt.ylabel("产品单价")

df_finshdata


# ### 四、进入分析


# 每条记录总金额：df["pro_mon"]=df["unitprice"]*df["quantity"]
# 总销售额：total_mon=df["pro_mon"].sum()
# 笔数：total_bishu=df["voiceNo"].unique
# 连带率：df["quantity"].sum()
# 件数：total_jianshu=df["quantity"].sum()
# 总客户数：total_customer_count=df["customerId"].unique.count()
# 笔单价：total_mon/total_bishu
# 客单价：total_mon/total_customer_count

# 总销售额：
total_price=df_finshdata["总价"].sum()
total_price

# 去重后 的总笔数
total_order_count=len(df["发票号"].unique())

# 连带率，先求订单内所有商品的总件数
total_pro_count=df_finshdata["产品数量"].sum()

# 连带率，订单内所有商品的总件数 除以 总笔数
joint_rate=total_pro_count/total_order_count

joint_rate

#客单价 =总销售额 / 总客户数
customer_price=total_price/len(df["用户ID"].unique())

customer_price

# 相关分析1
df_finshdata.groupby("产品数量").mean()

# 相关分析2
cus_avg_price=df_finshdata.groupby("用户ID").mean()["总价"].head(10)

plt.figure(figsize=(20,8),dpi=80)
plt.grid()
plt.bar(cus_avg_price.index,cus_avg_price.values,width=0.5)

cus_avg_price


# ### 第二部分练习
# + 商品维度：商品的价格定位是高是低？哪种价位的商品卖得好？哪种价位的商品带来了实际上最多的销售额
# + 时间维度：各月/各日的销售情况是什么走势？可能受到了什么影响？
# + 区位维度：客户主要来自哪几个国家？哪个国家是境外主要市场？哪个国家的客户平均消费能力最强

df_finshdata

df_finshdata.set_index("发票日期",drop=True,inplace=True)

# 按月统计销售额
month_sum=df_finshdata["总价"].resample("M").sum()

plt.figure(figsize=(20,8),dpi=80)
plt.grid()
plt.plot(month_sum.index,month_sum.values)
plt.xticks(month_sum.index[::2],month_sum.index[::2])

# 按月统计销售额
day_sum=df_finshdata["总价"].resample("D").sum()

plt.figure(figsize=(20,8),dpi=80)
plt.grid()
plt.plot(day_sum.index,day_sum.values)
plt.xticks(day_sum.index[::60],day_sum.index[::60])

# 哪种商品 卖的好
pro_cunt=df_finshdata.groupby("产品编码").sum()["产品数量"]
pro_cunt.sort_values(ascending=False).head()

# plt.figure(figsize=(20,8),dpi=80)
# plt.grid()
# plt.bar(pro_cunt.index,pro_cunt.values,width=0.5)
# # plt.xticks(day_sum.index[::60],day_sum.index[::60])

#提取 时间年月日
# df_finshdata["day"]=df_finshdata["发票日期"].dt.date

# 按月分组求每个月总销售额
# month_price=df_finshdata.groupby("month").sum()["总价"]

df_finshdata.sort_values(by="总价",ascending=False)

# 按国家 分组销售额占比分析
sum_p=df_finshdata.sum()
df_finshdata.groupby("国家").sum()["总价"].apply(lambda x:x/sum_p)

#客户平均消费
df_finshdata

# 按国家的客户平均消费能力
df_finshdata.groupby(["国家","用户ID"]).sum()[["总价"]]
df_finshdata.groupby("国家").sum()["总价"].sort_values(ascending=False).head(10)

df11=pd.DataFrame( {
    "title":["Toy","Juman","Grumpier","Waiting","Father"]
    ,"genres":["Animation|Children's|Comedy","Adventure|Children's|Fantasy","Comedy|Romance","Comedy|Drama","Comedy"]
})
df11

# 获取genres字段所有值，然后切割放在list，使用set对list去重返回，得到 不重复的类型名称
import collections

# def func(x):
#    print( x.split("|"))
genres=df11['genres']
genres_list=genre.str.split('|').tolist()
genres_set=collections.Counter([i for item in genre_list for i in item])
genres_set
