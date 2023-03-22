#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 23:14:03 2023

@author: cangyimeng
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

# 将股票交易日和财报公布日结合起来
def combine_date(trade_date, date):
    '''
    Parameters
    ----------
    trade_date : Array of datetime64[ns]
        日度交易日期
    date : Dataframe
        索引为股票的公告期，值为股票财报的发布日
    Returns
    -------
    available_date : Dataframe
        交易日为索引，交易日对应可获得的财报期为值

    '''

    def get_date(trade_date, x):
        result = [np.nan] * len(trade_date)
        x = x.dropna()
        trade_date_idx = 0
        for i in range(len(x)):  
            if i < len(x) - 1:
                while trade_date_idx < len(trade_date) and trade_date[trade_date_idx] <= x[i + 1]:
                    if trade_date[trade_date_idx] >= x[i]:
                        result[trade_date_idx] = x.index[i]
                    trade_date_idx += 1
            else:
                while trade_date_idx < len(trade_date):
                    if trade_date[trade_date_idx] >= x[i]:
                        result[trade_date_idx] = x.index[i]
                    trade_date_idx += 1
                
        return result

    tqdm.pandas(desc='apply')
    available_date = date.progress_apply(lambda x: get_date(trade_date, x))
    available_date.index = trade_date           
    return available_date


     

def get_available_data(available_date, data):
    '''
    Parameters
    ----------
    available_date : Dataframe
        交易日为索引，交易日对应可获得的财报期为值
    data : Dataframe
        股票财报数据，索引为报告期
    Returns
    -------
    available_data : Dataframe
        索引为交易日，值为交易日可获得的最新财务数据

    '''
    available_data = pd.DataFrame(index=available_date.index, columns=available_date.columns)
    
    columns = available_date.columns
    len_columns = len(columns)

    for j, row in tqdm(available_date.iterrows(), total=len(available_date)):
        for i in range(len_columns):
            date = row[columns[i]]
            stock = columns[i]
            if not pd.isna(date) and stock in data.columns:
                available_data.at[j, stock] = data.at[date, stock]
    return available_data
     
            
            
            