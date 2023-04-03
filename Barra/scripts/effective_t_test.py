#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30 20:13:52 2023

@author: cangyimeng + dingyuan
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
folder_path = '/Users/caterpillar/Desktop/indicators/Barra/Barra/data/raw_data/日个股回报率/'
file_names = glob.glob(folder_path + '*.csv')

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
path = '/Users/caterpillar/Desktop/indicators/Barra/Barra/data/standardized_risk_factors/'
factors_dict['beta'] = pd.read_csv(path+'beta_standardized.csv')
factors_dict['momentum'] = pd.read_csv(path+'RSTR_standardized.csv')
factors_dict['size'] = pd.read_csv(path+'LNCAP_standardized.csv')
factors_dict['earnings'] = pd.read_csv(path+'earnings_yield_factor_standardized.csv')
factors_dict['volatility'] = pd.read_csv(path+'volatility_factor_standardized.csv')
factors_dict['growth'] = pd.read_csv(path+'growth_factor_standardized.csv')
factors_dict['value'] = pd.read_csv(path+'BTOP_standardized.csv')
factors_dict['leverage'] = pd.read_csv(path+'leverage_factor_standardized.csv')
factors_dict['liquidity'] = pd.read_csv(path+'liquidity_factor_standardized.csv')



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

file_names = glob.glob('/Users/caterpillar/Desktop/indicators/Barra/Barra/data/industry_factors/*.csv')
for file_name in file_names:
    key = file_name.split('/')[-1].split('.')[0]
    factors_dict[key] = pd.read_csv(file_name)
    factors_dict[key] = factors_dict[key].rename(columns={'Unnamed: 0' : 'Trddt'})
    factors_dict[key] = factors_dict[key].set_index('Trddt')
    factors_dict[key].index = pd.to_datetime(factors_dict[key].index)
    factors_dict[key] = factors_dict[key].loc[trade_date, stock_list]
    

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

# Initialize dictionaries to store WLS t-test values and parameters
WLS_params = {}
WLS_t_test = {}

for i in tqdm(range(len(trade_date))):
    date = trade_date[i]
    wls_model = sm.WLS(R.loc[date, :], X[date], weights=weight.loc[date, :])
    output = wls_model.fit()
    WLS_params[date] = output.params
    WLS_t_test[date] = output.tvalues

# Calculate mean absolute t-test and proportion of absolute t-test values greater than 2
mean_abs_t_test = {}
proportion_t_test_gt_2 = {}
'''
for factor in X[date].columns:
    t_values = [WLS_t_test[date][factor] for date in WLS_t_test]
    mean_abs_t_test[factor] = np.mean(np.abs(t_values))
    proportion_t_test_gt_2[factor] = np.mean(np.array(t_values) > 2)
'''


# Calculate mean absolute t-test and proportion of absolute t-test values greater than 2
mean_abs_t_test = {}
proportion_t_test_gt_2 = {}

for factor in X[date].columns:
    t_values = [WLS_t_test[date][factor] for date in WLS_t_test if not np.isnan(WLS_t_test[date][factor])]
    if len(t_values) > 0:
        mean_abs_t_test[factor] = np.mean(np.abs(t_values))
        proportion_t_test_gt_2[factor] = np.mean(np.array(t_values) > 2)
    else:
        mean_abs_t_test[factor] = np.nan
        proportion_t_test_gt_2[factor] = np.nan

# Save mean_abs_t_test and proportion_t_test_gt_2 in a separate CSV file
t_test_results = pd.DataFrame({"Factor": list(mean_abs_t_test.keys()),
                               "Mean_Abs_T_Test": list(mean_abs_t_test.values()),
                               "Proportion_Abs_T_Test_GT_2": list(proportion_t_test_gt_2.values())})
t_test_results.to_csv('/Users/caterpillar/Desktop/indicators/Barra/Barra/data/factor_effective_tests/T_Test_Results.csv', index=False)