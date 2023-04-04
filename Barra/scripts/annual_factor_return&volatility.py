import statsmodels.api as sm
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

factor_returns = pd.read_csv('data/WLS_result.csv')

# 计算年化因子收益率
num_days = len(factor_returns)
continuous_growth_rates = 1 + factor_returns
annual_factor_return = (continuous_growth_rates.prod()) ** (252 / num_days) - 1

print("年化因子收益率:", annual_factor_return)

# 将整个时间段划分为4个接近相等的时间段
num_periods = 4
period_length = len(factor_returns) // num_periods

# 计算每个因子的年化波动率
annual_volatility_dict = {}

for factor in factor_returns.columns:
    factor_data = factor_returns[factor]
    period_returns = []

    for i in range(num_periods):
        start_idx = i * period_length
        end_idx = (i + 1) * period_length if i < num_periods - 1 else len(factor_data)
        period_data = factor_data[start_idx:end_idx]

        # 计算每个时间段的收益率
        continuous_growth_rates = 1 + period_data
        period_return = (continuous_growth_rates.prod()) ** (252 / len(period_data)) - 1
        period_returns.append(period_return)

    # 计算年化波动率
    annual_volatility = np.std(period_returns, ddof=1)
    annual_volatility_dict[factor] = annual_volatility

# 打印每个因子的年化波动率
for factor, volatility in annual_volatility_dict.items():
    print(f"{factor} 年化波动率: {volatility}")

index =['volatility']
volatility = pd.DataFrame(annual_volatility_dict, index=index)
annual_factor_return.to_csv('data/annual_factor_return.csv')
volatility.to_csv('data/volatility.csv')
