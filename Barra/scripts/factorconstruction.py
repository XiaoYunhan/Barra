#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 17:11:45 2023

@author: cangyimeng
"""

import pandas as pd
import numpy as np
import glob
import datacleaner as dc
from tqdm import tqdm
import statsmodels.api as sm

def ewma_beta(returns, benchmark_returns, halflife):
    X = sm.add_constant(benchmark_returns)
    model = sm.WLS(returns, X, weights=np.power(2,-np.arange(len(returns)) / halflife))
    results = model.fit()
    return results.params[300]

def ewma(returns, benchmark_returns, halflife):
    X = sm.add_constant(benchmark_returns)
    model = sm.WLS(returns, X, weights=np.power(2,-np.arange(len(returns)) / halflife))
    results = model.fit()
    return results

def ewma_hsigma(returns, benchmark_returns, halflife):
    weights = np.power(2, -np.arange(len(returns)) / halflife)
    X = sm.add_constant(benchmark_returns)
    model = sm.WLS(returns, X, weights=weights)
    results = model.fit()
    residuals = results.resid
    hsigma = np.sqrt(np.dot(weights, residuals**2) / np.sum(weights))
    return hsigma

def ewma_std(returns, halflife):
    # 计算个股收益率的差值（减去均值）
    mean_returns = np.mean(returns)
    returns_diff = returns - mean_returns
    
    # 计算权重
    weights = np.power(2, -np.arange(len(returns)) / halflife)
    
    # 归一化权重
    weights = weights / np.sum(weights)
    
    # 计算加权平方差
    weighted_variance = np.sum(weights * returns_diff ** 2)
    
    # 计算标准差（DASTD）
    dastd = np.sqrt(weighted_variance)
    return dastd

def cmra(monthly_returns):
    cumulative_log_returns = np.cumsum(np.log(1 + monthly_returns))
    max_return = np.max(cumulative_log_returns)
    min_return = np.min(cumulative_log_returns)
    cmra_value = max_return - min_return
    return cmra_value

def daily_to_monthly_returns(daily_returns, dates):
    daily_returns_series = pd.Series(daily_returns, index=dates)
    monthly_returns_series = daily_returns_series.resample('M').apply(lambda x: (1 + x).prod() - 1)
    return monthly_returns_series

'''
part1: 数据导入&整理，所有数据都为矩阵形式
'''

'''
财报数据
'''
report_date = pd.read_excel('财报公布日期/IAR_Rept.xlsx')

report_date = report_date.iloc[2:,]
report_date = report_date.loc[:,['Stkcd', 'Reptyp', 'Accper', 'Annodt']]
# Stkcd [证券代码] - 以沪深交易所公布的最新证券代码为准
# Reptyp [报告类型] - 1=第一季度季报，2=中报，3=第三季度季报，4=年报
# Accper [统计截止日期] - 以YYYY-MM-DD列示，部分缺少在相应位置上以00表示，如1993年12月某日表示为1993-12-00
# Annodt [报告公布日期] - 以YYYY-MM-DD列示，部分缺少在相应位置上以00表示，如1993年12月某日表示为1993-12-00

# 剔除缺少日期的股票
report_date = report_date.dropna(subset=['Annodt'])
report_date = report_date[~ report_date['Annodt'].str.endswith('-00')]
report_date['Stkcd'] = report_date['Stkcd'].astype(int)
report_date = report_date.drop_duplicates()

# 结合可用日期，仅保留合并报表数据
report_cash = pd.read_csv('现金流量表/FS_Comscfd.csv')
report_asset = pd.read_csv('资产负债表/FS_Combas.csv')
report_profit = pd.read_csv('利润表/FS_Comins.csv')

report_cash = pd.merge(report_date, report_cash, on = ['Stkcd', 'Accper'], how = 'right')
report_cash = report_cash.dropna(subset=['Annodt'])
report_cash = report_cash[~report_cash['Typrep'].str.contains('B')]
report_cash.Accper = pd.to_datetime(report_cash.Accper)
report_cash.Annodt = pd.to_datetime(report_cash.Annodt)

report_asset = pd.merge(report_date, report_asset, on = ['Stkcd', 'Accper'], how = 'right')
report_asset = report_asset.dropna(subset=['Annodt'])
report_asset = report_asset[~report_asset['Typrep'].str.contains('B')]
report_asset.Accper = pd.to_datetime(report_asset.Accper)
report_asset.Annodt = pd.to_datetime(report_asset.Annodt)

report_profit = pd.merge(report_date, report_profit, on = ['Stkcd', 'Accper'], how = 'right')
report_profit = report_profit.dropna(subset=['Annodt'])
report_profit = report_profit[~report_profit['Typrep'].str.contains('B')]
report_profit.Accper = pd.to_datetime(report_profit.Accper)
report_profit.Annodt = pd.to_datetime(report_profit.Annodt)

report_date.Accper = pd.to_datetime(report_date.Accper)
report_date.Annodt = pd.to_datetime(report_date.Annodt)
report_date = pd.pivot(report_date, index = 'Accper', columns = 'Stkcd', values = 'Annodt')

# 个股净利润 B002000000
stock_net_profit = report_profit.pivot_table(index='Accper', columns = 'Stkcd', values = 'B002000000')

# 归母净利润
stock_par_profit = report_profit.pivot_table(index='Accper', columns = 'Stkcd', values = 'B002000101')

# 营业总收入
stock_income = report_profit.pivot_table(index='Accper', columns = 'Stkcd', values = 'B001100000')

# 总权益(账面权益)
stock_total_shares = report_asset.pivot_table(index='Accper', columns = 'Stkcd', values = 'A003000000')

# 总负债
stock_debt = report_asset.pivot_table(index='Accper', columns = 'Stkcd', values = 'A002000000')

# 长期负债
stock_long_debt = report_asset.pivot_table(index='Accper', columns = 'Stkcd', values = 'A002206000')

# 总资产
stock_asset = report_asset.pivot_table(index='Accper', columns = 'Stkcd', values = 'A001000000')

# 现金收益
stock_cash_income = report_cash.pivot_table(index='Accper', columns = 'Stkcd', values = 'C001000000')




'''
股票交易数据
'''
# 导入股票日数据
file_names = glob.glob('日个股回报率/*.csv')
stock_prices = pd.concat((pd.read_csv(file) for file in file_names), ignore_index=True)

# 保留A股、创业板、科创板股票
stock_prices = stock_prices[stock_prices['Markettype'].isin([1, 4, 16, 32])]
stock_prices.Trddt = pd.to_datetime(stock_prices.Trddt)

# 收益率
stock_returns = stock_prices.pivot_table(index='Trddt', columns='Stkcd', values='Dretwd')
# stock_returns = stock_returns.fillna(0)

# 总市值
stock_mkt_cap = stock_prices.pivot_table(index='Trddt', columns='Stkcd', values='Dsmvtll')
# stock_mkt_cap = stock_mkt_cap.fillna(0)

# 成交量
stock_volume = stock_prices.pivot_table(index='Trddt', columns='Stkcd', values='Dnshrtrd')
# stock_volume = stock_volume.fillna(0)

# 流通股本
# Dsmvosd [日个股流通市值] - 计算公式为：个股的流通股数与收盘价的乘积，A股以人民币元计，上海B股以美元计，深圳B股以港币计，注意单位是千
stock_prices['流通股本'] = stock_prices.Dsmvosd / stock_prices.Clsprc
stock_shares = stock_prices.pivot_table(index='Trddt', columns='Stkcd', values='流通股本')
# stock_shares = stock_shares.fillna(0)

# 股价
stock_price_mat = stock_prices.pivot_table(index='Trddt', columns='Stkcd', values='Clsprc')

'''
股票财务数据
'''
file_names = glob.glob('个股财务/*.csv')
stock_financial_index = pd.concat((pd.read_csv(file) for file in file_names), ignore_index=True)
stock_financial_index = stock_financial_index.rename(columns = {'TradingDate':'Trddt', 'Symbol':'Stkcd'})
stock_financial_index['Trddt'] = pd.to_datetime(stock_financial_index['Trddt'])
# PE [市盈率] - 市盈率＝股票市总值/最近四个季度的归属母公司的净利润之和
stock_PE = stock_financial_index.pivot_table(index='Trddt', columns='Stkcd', values='PE')

# PB [市净率] - 市净率＝股票市值/净资产。净资产为最新定期报告公布的净资产。
stock_PB = stock_financial_index.pivot_table(index='Trddt', columns='Stkcd', values='PB')

# PCF [市现率] - 市现率＝股票市值/去年经营现金流量净额。
stock_PCF = stock_financial_index.pivot_table(index='Trddt', columns='Stkcd', values='PCF')

# PS [市销率] - 市销率＝股票市值/去年营业收入。
stock_PS = stock_financial_index.pivot_table(index='Trddt', columns='Stkcd', values='PS')

'''
股票一致预期数据
'''
file_names = glob.glob('滚动一致预期/*.csv')
stock_expect_ttm = pd.concat((pd.read_csv(file) for file in file_names), ignore_index=True)
stock_expect_ttm = stock_expect_ttm.rename(columns = {'ForecastDate':'Trddt','Symbol' : 'Stkcd'})

# 一致预期每股收益
stock_EPS = stock_expect_ttm.pivot_table(index = 'Trddt', columns = 'Stkcd', values = 'EPS')
stock_EPS.index = pd.to_datetime(stock_EPS.index)


file_names = glob.glob('一致预期/*.csv')
stock_expect = pd.concat((pd.read_csv(file) for file in file_names), ignore_index=True)
stock_expect = stock_expect.rename(columns = {'ForecastDate':'Trddt','Symbol' : 'Stkcd'})
stock_expect['Trddt'] = pd.to_datetime(stock_expect['Trddt'])
stock_expect['ForecastYear'] = pd.to_datetime(stock_expect['ForecastYear'])
# stock_expect['year_gap'] = stock_expect['ForecastYear'].dt.year - stock_expect['Trddt'].dt.year
stock_expect = stock_expect[['Stkcd', 'Trddt', 'ForecastYear', 'NetProfit']]
stock_expect = stock_expect.sort_values(by = ['Trddt', 'ForecastYear'])

n = stock_expect.groupby(['Stkcd', 'Trddt']).count().reset_index()
n.columns = ['Stkcd', 'Trddt','x','y']
stock_expect = pd.merge(stock_expect, n, on = ['Stkcd', 'Trddt'], how = 'inner')
stock_expect = stock_expect[stock_expect['x'] == 3]
stock_expect = stock_expect[['Stkcd', 'Trddt', 'ForecastYear', 'NetProfit']]

# 导入沪深300指数收益率
index_300 = pd.read_csv('300日收益率.csv')
index_300['Trddt'] = pd.to_datetime(index_300['Trddt'])
index_300 = index_300.pivot_table(index='Trddt', columns='Indexcd', values='Retindex')
index_300.columns = ['hs300']

# 筛选股票池
stock_list = set(report_cash.Stkcd).intersection(set(report_asset.Stkcd))
stock_list = stock_list.intersection(set(report_profit.Stkcd))
stock_list = stock_list.intersection(set(stock_prices.Stkcd))
stock_list = stock_list.intersection(set(stock_income.columns))
stock_list = stock_list.intersection(set(stock_total_shares.columns))
stock_list = stock_list.intersection(set(stock_long_debt.columns))
stock_list = stock_list.intersection(set(stock_PE.columns))
stock_list = stock_list.intersection(set(stock_PB.columns))
stock_list = stock_list.intersection(set(stock_PCF.columns))
stock_list = stock_list.intersection(set(stock_EPS.columns))

trade_date = set(stock_volume.index).intersection(set(stock_PB.index))
trade_date = np.sort(list(trade_date))

'''
行业分类
'''
stock_industry = pd.read_csv('行业分类/STK_IndustryClassAnl.csv')

# 使用申万分类P0211, P0218
stock_industry = stock_industry[stock_industry.IndustryClassificationID.isin(['P0211', 'P0218'])]
stock_industry = stock_industry[['EndDate', 'Symbol', 'IndustryCode1','IndustryName1']]
stock_industry.EndDate = pd.to_datetime(stock_industry.EndDate)

# 整合时间
trade_date_year = pd.DataFrame(trade_date)
trade_date_year['year'] = trade_date_year.iloc[:,0].dt.year
stock_industry['year'] = stock_industry['EndDate'].dt.year

stock_ind_2023 = stock_industry[stock_industry['year'] == 2022]
stock_ind_2023['year'] = 2023
stock_industry = pd.concat([stock_industry, stock_ind_2023])
stock_industry = pd.merge(trade_date_year, stock_industry, on = 'year', how = 'left')

unique_industries = stock_industry.IndustryCode1.unique()
unique_stock_codes = stock_industry.Symbol.unique()

stock_list = set(stock_list).intersection(set(unique_stock_codes))
stock_list = np.sort(list(stock_list))

# 合并时间
available_date = dc.combine_date(trade_date, report_date.loc[:,stock_list])
rpt_date = np.sort(available_date.stack().unique())


'''
part2: 因子计算，输出矩阵形式因子
'''

# stock_expect_date为交易日可获得的最新报告期及预测报告期
available_date_stack = available_date.stack().reset_index().dropna()
available_date_stack = available_date_stack.rename(columns = {'level_0': 'Trddt', 0: 'report_date'})
stock_expect_date = pd.merge(stock_expect, available_date_stack, on = ['Stkcd', 'Trddt'], how = 'inner')
stock_expect_date['year_gap'] = stock_expect_date['ForecastYear'].dt.year - stock_expect_date['Trddt'].dt.year




# EGIB 未来 3 年企业一致预期净利润增长率。
stock_expect_pro3 = pd.DataFrame(index = trade_date, columns = stock_list)
temp = stock_expect_date[stock_expect_date['year_gap'] == 2].pivot_table(index = 'Trddt', 
                                                                         columns = 'Stkcd',
                                                                         values = 'NetProfit')

stock_expect_pro3.loc[temp.index, temp.columns] = temp

available_Pro = dc.get_available_data(available_date, stock_net_profit)
EGIB = stock_expect_pro3.loc[trade_date,stock_list]/available_Pro.loc[trade_date,stock_list] - 1


# EGIB_S 未来 1 年企业一致预期净利润增长率。
stock_expect_pro1 = pd.DataFrame(index = trade_date, columns = stock_list)
temp = stock_expect_date[stock_expect_date['year_gap'] == 0].pivot_table(index = 'Trddt', 
                                                                         columns = 'Stkcd',
                                                                         values = 'NetProfit')
stock_expect_pro1.loc[temp.index, temp.columns] = temp
EGIB_S = stock_expect_pro1.loc[trade_date,stock_list]/available_Pro.loc[trade_date,stock_list] - 1


# MLEV = (ME + LD) / ME ;其中ME 表示企业当前总市值，LD 表示企业长期负债。
available_LD = dc.get_available_data(available_date, stock_long_debt)
MLEV = (stock_mkt_cap.loc[trade_date,stock_list] + available_LD.loc[trade_date,stock_list]) / stock_mkt_cap.loc[trade_date,stock_list]

# DTOA = TD / TA ;其中 TD 表示总负债 TA 表示总资产。
DTOA = stock_debt.loc[:,stock_list]/stock_asset.loc[:,stock_list]
DTOA = dc.get_available_data(available_date, DTOA)

# BLEV = (BE + LD) / BE ;其中BE 表示企业账面权益，LD 表示企业长期负债。
BLEV = (stock_total_shares.loc[rpt_date,stock_list] + stock_long_debt.loc[rpt_date,stock_list]) / stock_total_shares.loc[rpt_date,stock_list]
BLEV = dc.get_available_data(available_date, BLEV)

# LNCAP 个股总市值对数值。
LNCAP = np.log(stock_mkt_cap.loc[trade_date,stock_list])

# EPIBS EPIBS = est _ eps / P ;其中est _ eps 为个股一致预期基本每股收益。
EPIBS = stock_EPS.loc[trade_date,stock_list]/stock_price_mat.loc[trade_date,stock_list]

# ETOP 市盈率倒数，市盈率＝股票市总值/最近四个季度的归属母公司的净利润之和
ETOP = 1/stock_PE.loc[trade_date,stock_list]

# # CETOP = 个股现金收益比股票价格。
# available_cash = dc.get_available_data(available_date, stock_cash_income)
# CETOP = available_cash.loc[trade_date,stock_list]/stock_price_mat.loc[trade_date,stock_list]

# CETOP = 市现率倒数，市现率＝股票市值/去年经营现金流量净额。
CETOP = 1/stock_PCF.loc[trade_date,stock_list]


# SGRO 过去 5 年企业营业总收入复合增长率。(简单起见，直接用20个季度算)
SGRO = (stock_income.pct_change(periods=20)+1)**(1/5) - 1
SGRO = SGRO.loc[rpt_date, stock_list]
SGRO = dc.get_available_data(available_date, SGRO)

# EGRO 过去 5 年企业归属母公司净利润复合增长率。
EGRO = (stock_par_profit.pct_change(periods=20)+1)**(1/5) - 1
EGRO = EGRO.loc[rpt_date, stock_list]
EGRO = dc.get_available_data(available_date, EGRO)

# # BTOP 计算企业总权益值除以当前市值。
# available_share = dc.get_available_data(available_date, stock_total_shares)
# BTOP = available_share.loc[trade_date,stock_list]/stock_mkt_cap.loc[trade_date,stock_list]

# BTOP：市净率倒数 市净率＝股票市值/净资产。净资产为最新定期报告公布的净资产。
BTOP = 1/stock_PB.loc[trade_date,stock_list]


#计算beta
# beta_values = {}
# for stock in stock_returns.columns:
#     stock_beta_values = []
#     for i in range(len(stock_returns) - 249):
#         window_stock_returns = stock_returns[stock].iloc[i:i+250]
#         window_hs300_returns = index_300.iloc[i:i+250]  
#         beta = ewma_beta(window_stock_returns, window_hs300_returns, 60)
#         stock_beta_values.append(beta)
#     beta_values[stock] = stock_beta_values
# beta_values = pd.DataFrame(beta_values)
# beta_values.index = stock_returns.iloc[249:1214,:].index

# #计算DATSD
# DATSD_values = {}
# for stock in stock_returns.columns:
#     stock_datsd_values = []
#     for i in range(len(stock_returns) - 249):
#         window_stock_returns = stock_returns[stock].iloc[i:i+250]
#         datsd = ewma_std(window_stock_returns, 40)
#         stock_datsd_values.append(datsd)
#     DATSD_values[stock] = stock_datsd_values
# DATSD_values = pd.DataFrame(DATSD_values)
# DATSD_values.index = stock_returns.iloc[249:1214,:].index

# #计算CMRA
# CMRA_values = {}
# for stock in stock_returns.columns:
#     stock_CMRA_values = []
#     a = daily_to_monthly_returns(stock_returns[stock], stock_returns.index)
#     for i in range(len(a)-12):
#         window_stock_returns = a[i:i+12]
#         CMRA = cmra(window_stock_returns)
#         stock_CMRA_values.append(CMRA)
#     CMRA_values[stock] = stock_CMRA_values
# CMRA_values = pd.DataFrame(CMRA_values)
# CMRA_values.index = a.iloc[12:61].index

# #计算Hsigma
# HSIGMA_values = {}
# for stock in stock_returns.columns:
#     stock_hsigma_values = []
#     for i in range(len(stock_returns) - 249):
#         window_stock_returns = stock_returns[stock].iloc[i:i+250]
#         window_hs300_returns = index_300.iloc[i:i+250]  
#         hsigma = ewma_hsigma(window_stock_returns, window_hs300_returns, 60)
#         stock_hsigma_values.append(hsigma)
#     HSIGMA_values[stock] = stock_hsigma_values
# HSIGMA_values = pd.DataFrame(HSIGMA_values)
# HSIGMA_values.index = stock_returns.iloc[249:1214,:].index

beta = pd.read_csv('beta.csv')
datsd = pd.read_csv('datsd.csv')
cmra = pd.read_csv('cmra.csv')
hsigma = pd.read_csv('hsigma.csv')

beta = beta.loc[trade_date,stock_list]
datsd= datsd.loc[trade_date,stock_list]
cmra = cmra.loc[trade_date,stock_list]
hsigma = hsigma.loc[trade_date,stock_list]

# STOM = ln(sum21 (Vt /St));其中Vt 表示当日成交量，St 表示流通股本。
STOM = stock_volume.loc[:,stock_list] / stock_shares.loc[:,stock_list]
STOM = np.log(STOM.rolling(window = 21).sum())


# STOQ = ln(1/T sumT exp(STOM));其中T=3。
STOQ = np.exp(STOM)
STOQ = np.log(STOQ.rolling(window = 3).sum()/3)

# STOA : T = 12
STOA = np.exp(STOM)
STOA = np.log(STOA.rolling(window = 12).sum()/12)

# RSTR = sum(21-521)w_t [ln (1+rt )] ;其中 T=500，L=21，收益率序列以半衰指数加权，半衰期为 120 日。
weight = np.array([0.5**((520 - i)/120) for i in range(1, 521)]).reshape(-1)
def cal_RSTR(stock_returns, weight, T = 500, L = 21):
    log_return = np.log(1+stock_returns)
    RSTR = log_return*np.nan
    for i in tqdm(range(len(log_return)-T-L)):
        df = log_return[i:i+T+L-1].to_numpy()
        df = pd.DataFrame(df*weight[:, np.newaxis])
        RSTR.iloc[i + T + L, :] = df[0:T].sum()
    return RSTR

RSTR = cal_RSTR(stock_returns, weight)


STOM = STOM.loc[trade_date,stock_list]
STOQ = STOQ.loc[trade_date,stock_list]
STOA = STOA.loc[trade_date,stock_list]
RSTR = RSTR.loc[trade_date,stock_list]



# 行业因子，共32个行业
industry_dfs = {}
for industry in tqdm(unique_industries):
    industry_dfs[industry] = pd.DataFrame(index=trade_date, columns=stock_list).fillna(0)


for industry, group in tqdm(stock_industry.groupby('IndustryCode1')):
    for _, row in group.iterrows():
        industry_dfs[industry].at[row['EndDate'], row['Symbol']] = 1
 
folder_path = "../data/industry_factors"
# 存为csv
for industry, industry_df in industry_dfs.items():
    
    file_name =  f"{folder_path}{industry}.csv"
    
    industry_df.to_csv(file_name)
    print('保存至'+file_name)


