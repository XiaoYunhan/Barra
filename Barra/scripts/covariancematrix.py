#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 20:00:50 2023

@author: cangyimeng
"""
import numpy as np
import pandas as pd

'''
2.3. 组合风险预测
sigma_p = np.sqrt(w.T @ (X @ F @ X.T + delta) @ w)
F 为因子收益率的协方差矩阵，也是预测目标
X 为当期因子暴露矩阵（N × K 矩阵，K为因子个数）
delta 为 N × N 对角阵，其对角线上的元素对应个股的特异性收益率的方差
'''

f = pd.read_csv('../data/WLS_result_new.csv')
f = f.set_index('Trddt')

# 利用日频率的历史因子收益率，计算日频率的因子收益率协方差矩阵 F
# 其中利用了 RiskMetrics 的加权移动平均(EWMA)方法
def Barra_cov(f, q = 15):
    T = len(f)
    
    Gamma = {}
    
    Gamma[0] = 0
    for t in range(T):
        ft = pd.DataFrame(f.iloc[t, :])
        Gamma[0] += ft @ ft.T
        
        
    for i in range(1, q+1):
        for t in range(0, T-i):    
            Gamma[i] += f.iloc[t, :] @ f.iloc[t+i, :].T
    
    Gamma = np.matrix(Gamma)/T
    
    V_f = Gamma[0]
    for i in range(1, q+1):
        V_f += (1-i/(1+q)) * Gamma[i] @ Gamma[i].T
        
    return 22*V_f

Barra_cov(f)



tau = 90
h = 252

w = 0.5**(1/tau)
weight = [w**i for i in range(h,-1, -1)]

f_bar = f.rolling(window = h+1, min_periods = h+1).apply(lambda x: (x*weight).sum()/np.sum(weight))

f1 = f-f_bar
f1 = f1.dropna()

f2 = {}

for t in range(len(f1.index)):
    date = f1.index[t]
    f2[date] = np.matrix(f1.loc[date, :]).T @ np.matrix(f1.loc[date, :])
   
f3 = {}
for t in range(len(f1.index)-h):
    date1 = f1.index[t]
    date2 = f1.index[t+h]
    f3[date2] = f2[date1]*weight[0]
    for i in range(t+1, t+h+1):
        date3 = f1.index[i]
        f3[date2] += f2[date3]*weight[i-t]
    
    f3[date2] = f3[date2]/np.sum(weight)


delta = 1
f1_1 = f.shift(delta) - f_bar
f1_1 = f1_1.dropna()

f2_1_1 = {}
for t in range(len(f1_1.index)):
    date = f1_1.index[t]
    f2_1_1[date] = np.matrix(f1_1.loc[date, :]).T @ np.matrix(f1.loc[date, :])
   

   
f3_1_1 = {}
for t in range(len(f1_1.index)-h+delta):
    date1 = f1_1.index[t]
    date2 = f1_1.index[t+h-delta]
    f3_1_1[date2] = f2_1_1[date1]*weight[delta]
    for i in range(t+1, t+h+1-delta):
        date3 = f1_1.index[i]
        f3_1_1[date2] += f2_1_1[date3]*weight[i-t]
    
    f3_1_1[date2] = f3_1_1[date2]/np.sum(weight[delta:])



delta = 2
f1_2 = f.shift(delta) - f_bar
f1_2 = f1_2.dropna()

f2_1_2 = {}
for t in range(len(f1_2.index)):
    date = f1_2.index[t]
    f2_1_2[date] = np.matrix(f1_2.loc[date, :]).T @ np.matrix(f1.loc[date, :])
   

   
f3_1_2 = {}
for t in range(len(f1_2.index)-h+delta):
    date1 = f1_2.index[t]
    date2 = f1_2.index[t+h-delta]
    f3_1_2[date2] = f2_1_2[date1]*weight[delta]
    for i in range(t+1, t+h+1-delta):
        date3 = f1_2.index[i]
        f3_1_2[date2] += f2_1_2[date3]*weight[i-t]
    
    f3_1_2[date2] = f3_1_2[date2]/np.sum(weight[delta:])


delta = 1
f1_1 = f.shift(delta) - f_bar
f1_1 = f1_1.dropna()

f2_2_1 = {}
for t in range(len(f1_1.index)):
    date = f1_1.index[t]
    f2_2_1[date] = np.matrix(f1.loc[date, :]).T @ np.matrix(f1_1.loc[date, :])
   

   
f3_2_1 = {}
for t in range(len(f1_1.index)-h+delta):
    date1 = f1_1.index[t]
    date2 = f1_1.index[t+h-delta]
    f3_2_1[date2] = f2_2_1[date1]*weight[delta]
    for i in range(t+1, t+h+1-delta):
        date3 = f1_1.index[i]
        f3_2_1[date2] += f2_2_1[date3]*weight[i-t]
    
    f3_2_1[date2] = f3_2_1[date2]/np.sum(weight[delta:])



delta = 2
f1_2 = f.shift(delta) - f_bar
f1_2 = f1_2.dropna()

f2_2_2 = {}
for t in range(len(f1_2.index)):
    date = f1_2.index[t]
    f2_2_2[date] = np.matrix(f1.loc[date, :]).T @ np.matrix(f1_2.loc[date, :])
   

   
f3_2_2 = {}
for t in range(len(f1_2.index)-h+delta):
    date1 = f1_2.index[t]
    date2 = f1_2.index[t+h-delta]
    f3_2_2[date2] = f2_2_2[date1]*weight[delta]
    for i in range(t+1, t+h+1-delta):
        date3 = f1_2.index[i]
        f3_2_2[date2] += f2_2_2[date3]*weight[i-t]
    
    f3_2_2[date2] = f3_2_2[date2]/np.sum(weight[delta:])


cov_NW = {}
for i in range(len(f3.keys())):
    date = list(f3.keys())[i]
    cov_NW[date] = 21*f3[date]+(2/3)*(f3_2_1[date] + f3_1_1[date])+(1/3)*(f3_2_2[date] + f3_1_2[date])



np.save('../data/factor_cov.npy', cov_NW)

u = pd.read_csv('../data/WLS_resid.csv')
u = u.set_index('Trddt')
w = 0.5**(1/tau)
weight = [w**i for i in range(h,-1, -1)]

def weighted_sigma(u, weight):
    h = 252
    u_bar = u.rolling(window = h+1, min_periods = h+1).apply(lambda x: (x*weight).sum()/np.sum(weight))





f1 = f-f_bar
f1 = f1.dropna()

f2 = {}

for t in range(len(f1.index)):
    date = f1.index[t]
    f2[date] = np.matrix(f1.loc[date, :]).T @ np.matrix(f1.loc[date, :])
   
f3 = {}
for t in range(len(f1.index)-h):
    date1 = f1.index[t]
    date2 = f1.index[t+h]
    f3[date2] = f2[date1]*weight[0]
    for i in range(t+1, t+h+1):
        date3 = f1.index[i]
        f3[date2] += f2[date3]*weight[i-t]
    
    f3[date2] = f3[date2]/np.sum(weight)


delta = 1
f1_1 = f.shift(delta) - f_bar
f1_1 = f1_1.dropna()

f2_1_1 = {}
for t in range(len(f1_1.index)):
    date = f1_1.index[t]
    f2_1_1[date] = np.matrix(f1_1.loc[date, :]).T @ np.matrix(f1.loc[date, :])
   

   
f3_1_1 = {}
for t in range(len(f1_1.index)-h+delta):
    date1 = f1_1.index[t]
    date2 = f1_1.index[t+h-delta]
    f3_1_1[date2] = f2_1_1[date1]*weight[delta]
    for i in range(t+1, t+h+1-delta):
        date3 = f1_1.index[i]
        f3_1_1[date2] += f2_1_1[date3]*weight[i-t]
    
    f3_1_1[date2] = f3_1_1[date2]/np.sum(weight[delta:])



delta = 2
f1_2 = f.shift(delta) - f_bar
f1_2 = f1_2.dropna()

f2_1_2 = {}
for t in range(len(f1_2.index)):
    date = f1_2.index[t]
    f2_1_2[date] = np.matrix(f1_2.loc[date, :]).T @ np.matrix(f1.loc[date, :])
   

   
f3_1_2 = {}
for t in range(len(f1_2.index)-h+delta):
    date1 = f1_2.index[t]
    date2 = f1_2.index[t+h-delta]
    f3_1_2[date2] = f2_1_2[date1]*weight[delta]
    for i in range(t+1, t+h+1-delta):
        date3 = f1_2.index[i]
        f3_1_2[date2] += f2_1_2[date3]*weight[i-t]
    
    f3_1_2[date2] = f3_1_2[date2]/np.sum(weight[delta:])


delta = 1
f1_1 = f.shift(delta) - f_bar
f1_1 = f1_1.dropna()

f2_2_1 = {}
for t in range(len(f1_1.index)):
    date = f1_1.index[t]
    f2_2_1[date] = np.matrix(f1.loc[date, :]).T @ np.matrix(f1_1.loc[date, :])
   

   
f3_2_1 = {}
for t in range(len(f1_1.index)-h+delta):
    date1 = f1_1.index[t]
    date2 = f1_1.index[t+h-delta]
    f3_2_1[date2] = f2_2_1[date1]*weight[delta]
    for i in range(t+1, t+h+1-delta):
        date3 = f1_1.index[i]
        f3_2_1[date2] += f2_2_1[date3]*weight[i-t]
    
    f3_2_1[date2] = f3_2_1[date2]/np.sum(weight[delta:])



delta = 2
f1_2 = f.shift(delta) - f_bar
f1_2 = f1_2.dropna()

f2_2_2 = {}
for t in range(len(f1_2.index)):
    date = f1_2.index[t]
    f2_2_2[date] = np.matrix(f1.loc[date, :]).T @ np.matrix(f1_2.loc[date, :])
   

   
f3_2_2 = {}
for t in range(len(f1_2.index)-h+delta):
    date1 = f1_2.index[t]
    date2 = f1_2.index[t+h-delta]
    f3_2_2[date2] = f2_2_2[date1]*weight[delta]
    for i in range(t+1, t+h+1-delta):
        date3 = f1_2.index[i]
        f3_2_2[date2] += f2_2_2[date3]*weight[i-t]
    
    f3_2_2[date2] = f3_2_2[date2]/np.sum(weight[delta:])


cov_NW = {}
for i in range(len(f3.keys())):
    date = list(f3.keys())[i]
    cov_NW[date] = 21*f3[date]+(2/3)*(f3_2_1[date] + f3_1_1[date])+(1/3)*(f3_2_2[date] + f3_1_2[date])


    








