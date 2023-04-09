# %%
import pandas as pd
import glob
import numpy as np
from scipy.optimize import NonlinearConstraint

# 导入股票收益率数据
file_names = glob.glob('../data/raw_data/日个股回报率/*.csv')
print(file_names)
stock_prices = pd.concat((pd.read_csv(file) for file in file_names), ignore_index=True)

# 保留A股、创业板、科创板股票
stock_prices = stock_prices[stock_prices['Markettype'].isin([1, 4, 16, 32])]
stock_prices.Trddt = pd.to_datetime(stock_prices.Trddt)

# 收益率
stock_returns = stock_prices.pivot_table(index='Trddt', columns='Stkcd', values='Dretwd')

# %%
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

# %%
stock_returns = stock_returns.loc[trade_date, stock_list.astype(int)]
print(stock_returns)

# %%
# 目标函数
def expReturn(w, returns):
    print(f"w in expReturn: {w}")
    ret = np.dot(w, returns)
    print(f"ret in expReturn: {ret}")
    return -ret

# %%
# w_bench 
hs300weights = pd.read_csv('../data/hs300_weights.csv')
hs300weights = hs300weights.loc[:,['Stkcd','Enddt','Weight']]

hs300weights = hs300weights.pivot(index='Enddt', columns='Stkcd', values='Weight').fillna(0)
hs300weights.index = pd.to_datetime(hs300weights.index)
hs300weights = hs300weights.loc[trade_date,:]
new_hs300weights = hs300weights.reindex(columns=stock_list.astype(int), fill_value=0)

# 将数据框重新索引为trade_date, stock_list
new_hs300weights = new_hs300weights.loc[trade_date, stock_list.astype(int)]
print(new_hs300weights)

# %%
def weightConstraint_size(weight, weight_benchmark, risk_factor_exposure):
    return (np.matrix(weight).T - np.matrix(weight_benchmark).T) @ np.matrix(risk_factor_exposure)

def weightConstraint_beta(weight, weight_benchmark, risk_factor_exposure):
    return (np.matrix(weight).T - np.matrix(weight_benchmark).T) @ np.matrix(risk_factor_exposure)

def weightConstraint_momentum(weight, weight_benchmark, risk_factor_exposure):
    return (np.matrix(weight).T - np.matrix(weight_benchmark).T) @ np.matrix(risk_factor_exposure)

def weightConstraint_earnings(weight, weight_benchmark, risk_factor_exposure):
    return (np.matrix(weight).T - np.matrix(weight_benchmark).T) @ np.matrix(risk_factor_exposure)

def weightConstraint_volatility(weight, weight_benchmark, risk_factor_exposure):
    return (np.matrix(weight).T - np.matrix(weight_benchmark).T) @ np.matrix(risk_factor_exposure)

def weightConstraint_growth(weight, weight_benchmark, risk_factor_exposure):
    return (np.matrix(weight).T - np.matrix(weight_benchmark).T) @ np.matrix(risk_factor_exposure)

def weightConstraint_value(weight, weight_benchmark, risk_factor_exposure):
    return (np.matrix(weight).T - np.matrix(weight_benchmark).T) @ np.matrix(risk_factor_exposure)

def weightConstraint_leverage(weight, weight_benchmark, risk_factor_exposure):
    return (np.matrix(weight).T - np.matrix(weight_benchmark).T) @ np.matrix(risk_factor_exposure)

def weightConstraint_liquidity(weight, weight_benchmark, risk_factor_exposure):
    return (np.matrix(weight).T - np.matrix(weight_benchmark).T) @ np.matrix(risk_factor_exposure)

# %%
# 月末行业矩阵 >> H on page 18

factors_dict.keys()
industry_factors = {}

endOfMonth = pd.DataFrame(factors_dict['growth'].index)
endOfMonth.reset_index(inplace=True)
endOfMonth['year'] = pd.to_datetime(endOfMonth.Trddt).dt.year
endOfMonth['month'] = pd.to_datetime(endOfMonth.Trddt).dt.month

endOfMonth = endOfMonth.groupby(['year', 'month']).last()['Trddt']
# groupByYear = endOfMonth.groupby('year')

for day in endOfMonth:
    industry_factors[day] = pd.DataFrame()
    for key in factors_dict.keys():
        if key not in ['beta', 'momentum', 'size', 'earnings', 'volatility', 'growth', 'value', 'leverage', 'liquidity']:
            industry_factors[day][key] = factors_dict[key].loc[day,:].transpose()


# %%
industry_factors[endOfMonth.iloc[0]]

# %%
# 行业约束

# industry_factors[date] = (3770, 32)
# (964, 3770)

def industryConstraint(weights, weights_benchmark, industry_factors):
    # 1*N weights, 1*N weights_benchmark
    # return (np.matrix(weights).T - np.matrix(weights_benchmark).T) @ np.matrix(industry_factors)
    LHS = np.matrix(weights) @ np.matrix(industry_factors)
    RHS = np.matrix(weights_benchmark) @ np.matrix(industry_factors)
    return np.sum(LHS - RHS)

# %%
# w >= 0
non_neg_constraint = NonlinearConstraint(lambda x: x, lb=0, ub=np.inf)

# sum(w) = 1
sum_constraint = NonlinearConstraint(lambda x: np.sum(x) - 1, lb=0, ub=0)

# %%
# 优化
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

def optimizePortfolio(returns, date, factors_dict, factor, industry_factors, weight_benchmark):
    print(f"returns: {returns.head()}")
    print(f"date: {date}")
    print(f"factor: {factor}")
    first5_items = {k: industry_factors[k] for k in list(industry_factors)[:5]}
    print(f"industry_factors (first 5 items): {first5_items}")
    print(f"weight_benchmark: {weight_benchmark.head()}")
    if len(returns) == 0:
        raise ValueError("returns 参数的长度为零")
    N = len(returns)
    x0 = [1/N]*N
    beta = factors_dict['beta'].loc[date,:]
    momentum = factors_dict['momentum'].loc[date,:]
    size = factors_dict['size'].loc[date,:]
    earnings = factors_dict['earnings'].loc[date,:]
    volatility = factors_dict['volatility'].loc[date,:]
    growth = factors_dict['growth'].loc[date,:]
    value = factors_dict['value'].loc[date,:]
    leverage = factors_dict['leverage'].loc[date,:]
    liquidity = factors_dict['liquidity'].loc[date,:]

    cons_dict = {'size': NonlinearConstraint(lambda x: weightConstraint_size(x, weight_benchmark, size), lb=0, ub=0),
             'beta': NonlinearConstraint(lambda x: weightConstraint_beta(x, weight_benchmark, beta), lb=0, ub=0),
             'momentum': NonlinearConstraint(lambda x: weightConstraint_momentum(x, weight_benchmark, momentum), lb=0, ub=0),
             'earnings': NonlinearConstraint(lambda x: weightConstraint_earnings(x, weight_benchmark, earnings), lb=0, ub=0),
             'volatility': NonlinearConstraint(lambda x: weightConstraint_volatility(x, weight_benchmark, volatility), lb=0, ub=0),
             'growth': NonlinearConstraint(lambda x: weightConstraint_growth(x, weight_benchmark, growth), lb=0, ub=0),
             'value': NonlinearConstraint(lambda x: weightConstraint_value(x, weight_benchmark, value), lb=0, ub=0),
             'leverage': NonlinearConstraint(lambda x: weightConstraint_leverage(x, weight_benchmark, leverage), lb=0, ub=0),
             'liquidity': NonlinearConstraint(lambda x: weightConstraint_liquidity(x, weight_benchmark, liquidity), lb=0, ub=0)}
    
    industry_constraint = NonlinearConstraint(lambda x: industryConstraint(x, weight_benchmark, industry_factors[date]), lb=0, ub=0)
    new_dict = {key: cons_dict[key] for key in cons_dict.keys() if key != factor}
    constraint_list = [non_neg_constraint, sum_constraint, industry_constraint]

    for key in new_dict.keys():
        constraint_list.append(new_dict[key])

    constraints = tuple(constraint_list)
    returns = returns.fillna(0)
    returns_array = returns.to_numpy()
    print(f"x0: {x0}")
    print(f"returns shape: {returns.shape}")
    print(f"bounds: {[(0, 1) for _ in range(len(returns))]}")
    print(f"constraints: {constraints}")

    res = minimize(fun=expReturn,
               x0=x0,                # init_w,
               args=(returns_array,),
               bounds = [(0, 1) for _ in range(len(returns))],
               constraints=constraints,
               method='trust-constr',
               tol=1e-6,
               options={'maxiter': 100000})
    
    return res.x

# %%
weight = {}
for factor in ['beta', 'momentum', 'size', 'earnings', 'volatility', 'growth', 'value', 'leverage', 'liquidity']:
    weight[factor] = pd.DataFrame(index = endOfMonth, columns = stock_list)
    for date in endOfMonth:
        if len(stock_returns.loc[date, :]) == 0:
            print(f"Warning: Empty returns for date {date}")
            continue
        weight[factor].loc[date, :] = optimizePortfolio(stock_returns.loc[date,:], date, factors_dict, factor, industry_factors, new_hs300weights.loc[date,:])

print('Done!')

# %%
weight['volatility']


