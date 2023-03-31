#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 20:13:52 2023

@author: cangyimeng
"""

import statsmodels.api as sm
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

'''
对于N 只股票的组合而言，组合的收益率向量可以写成:
R = X@f + U
X 表示所有因子的载荷矩阵，由行业因子哑变量矩阵和公共因子载荷矩阵构成
f 表示所有因子的因子收益率
U 表示残差序列。

f的估计为：= np.inv(X.T @ W @ X) @ X.T @ W @ R
'''


'''
导入数据
'''
# 导入股票收益率数据
file_names = glob.glob('../data/raw_data/日个股回报率/*.csv')
stock_prices = pd.concat((pd.read_csv(file) for file in file_names), ignore_index=True)

# 保留A股、创业板、科创板股票
stock_prices = stock_prices[stock_prices['Markettype'].isin([1, 4, 16, 32])]
stock_prices.Trddt = pd.to_datetime(stock_prices.Trddt)

# 收益率
stock_returns = stock_prices.pivot_table(index='Trddt', columns='Stkcd', values='Dretwd')
stock_returns.columns = stock_returns.columns.astype('str')

# 总市值
stock_mkt_cap = stock_prices.pivot_table(index='Trddt', columns='Stkcd', values='Dsmvtll')
weight = 1/(stock_mkt_cap)
weight.columns = weight.columns.astype('str')
# 导入所有标准化因子数据和行业因子
factors_dict = {}
factors_dict['beta'] = pd.read_csv('../data/standardized_risk_factors/beta_standardized.csv')
factors_dict['momentum'] = pd.read_csv('../data/standardized_risk_factors/RSTR_standardized.csv')
factors_dict['size'] = pd.read_csv('../data/standardized_risk_factors/LNCAP_standardized.csv')
factors_dict['earnings'] = pd.read_csv('../data/standardized_risk_factors/earnings_yield_factor_standardized.csv')
factors_dict['volatility'] = pd.read_csv('../data/standardized_risk_factors/volatility_factor_standardized.csv')
factors_dict['growth'] = pd.read_csv('../data/standardized_risk_factors/growth_factor_standardized.csv')
factors_dict['value'] = pd.read_csv('../data/standardized_risk_factors/BTOP_standardized.csv')
factors_dict['leverage'] = pd.read_csv('../data/standardized_risk_factors/leverage_factor_standardized.csv')
factors_dict['liquidity'] = pd.read_csv('../data/standardized_risk_factors/liquidity_factor_standardized.csv')


for factor in factors_dict.keys():
    factors_dict[factor] = factors_dict[factor].rename(columns={'Unnamed: 0' : 'Trddt'})
    factors_dict[factor] = factors_dict[factor].set_index('Trddt')
    factors_dict[factor].index = pd.to_datetime(factors_dict[factor].index)


trade_date = factors_dict['growth'].index
stock_list = factors_dict['growth'].columns.values

for factor in factors_dict.keys():
    factors_dict[factor] = factors_dict[factor].loc[trade_date, stock_list]
    factors_dict[factor] = factors_dict[factor].fillna(0)
    # factors_dict[factor] = factors_dict[factor].dropna(axis = 0, how = 'all')

file_names = glob.glob('../data/industry_factors/*.csv')
for file_name in file_names:
    key = file_name.split('/')[-1].split('.')[0]
    factors_dict[key] = pd.read_csv(file_name)
    factors_dict[key] = factors_dict[key].rename(columns={'Unnamed: 0' : 'Trddt'})
    factors_dict[key] = factors_dict[key].set_index('Trddt')
    factors_dict[key].index = pd.to_datetime(factors_dict[key].index)
    factors_dict[key] = factors_dict[key].loc[trade_date, stock_list]
    
# for factor in factors_dict.keys():
#     factors_dict[factor] = factors_dict[factor].fillna(0)
    
stock_returns = stock_returns.loc[trade_date, stock_list]
stock_returns = stock_returns.fillna(0)
weight = weight.loc[trade_date, stock_list]
weight = weight.fillna(0)

R = stock_returns
X = {}

for i in tqdm(range(len(trade_date))):
    date = trade_date[i]
    X[date] = pd.concat((factors_dict[factor].loc[date,:] for factor in factors_dict.keys()), axis = 1)    
    X[date].columns = [factor for factor in factors_dict.keys()]

f = []
for i in tqdm(range(len(trade_date))):
    date = trade_date[i]
    wls_model = sm.WLS( R.loc[date, :],X[date], weights=weight.loc[date, :])    
    f.append(wls_model.fit().params)

f = pd.DataFrame(f)

f.to_csv('../data/WLS_result.csv')




