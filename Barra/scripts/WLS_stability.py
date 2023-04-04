import pandas as pd

# 读取CSV文件
df = pd.read_csv('../data/WLS_result.csv')

# 计算除第一列以外的每一列的标准差（波动率）
volatility = df.iloc[:, 1:].std()

# 除去第一列的所有数据除以对应列的波动率
df.iloc[:, 1:] = df.iloc[:, 1:].div(volatility)

# 保存新的CSV文件
df.to_csv('../data/WLS_stability.csv', index=False)
