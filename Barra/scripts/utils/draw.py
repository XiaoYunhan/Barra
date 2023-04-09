import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

'''
#折线图
'''
# 生成数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = y1 / y2

# 创建画布并设置分辨率
fig, ax1 = plt.subplots(dpi=150)

# 设置左坐标轴
ax1.plot(x, y1, label="波动率1", color="blue")
ax1.plot(x, y2, label="波动率2", color="green")
ax1.set_xlabel("X轴")
ax1.set_ylabel("波动率")

# 将左纵坐标轴的数据表示为百分数
formatter = FuncFormatter(lambda y, _: '{:.0%}'.format(y))
ax1.yaxis.set_major_formatter(formatter)

# 设置右坐标轴
ax2 = ax1.twinx()
ax2.plot(x, y3, label="比值", color="red")
ax2.set_ylabel("比值")

# 将图例移到x坐标轴下方
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

# 设置标题并显示图像
plt.title("示例折线图")
plt.show()


'''
#柱状图
'''

# 生成数据
industries = [
    "行业1", "行业2", "行业3", "行业4", "行业5", "行业6", "行业7", "行业8", "行业9", "行业10",
    "行业11", "行业12", "行业13", "行业14", "行业15", "行业16", "行业17", "行业18", "行业19", "行业20",
    "行业21", "行业22", "行业23", "行业24", "行业25", "行业26", "行业27", "行业28"
]
weights = np.random.rand(len(industries)) * 0.25

# 创建画布并设置分辨率
fig, ax = plt.subplots(dpi=150)

# 绘制柱状图
ax.bar(industries, weights)

# 设置坐标轴
ax.set_xlabel("行业")
ax.set_ylabel("权重")
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
plt.xticks(rotation=90)

# 设置标题
plt.title("沪深300行业权重柱状图")

# 将图例移到x坐标轴下方
ax.legend(["沪深300行业权重"], loc='upper center', bbox_to_anchor=(0.5, -0.15))

# 显示图像
plt.show()


'''
#风格因子收益归因分析
'''


# 生成数据
style_factors = ["因子1", "因子2", "因子3", "因子4", "因子5"]
num_factors = len(style_factors)
num_portfolios = 3

# 随机生成敞口数据
np.random.seed(42)
exposures = np.random.uniform(-3, 1, size=(num_factors, num_portfolios))

# 创建画布并设置分辨率
fig, ax = plt.subplots(dpi=150)

# 设置柱状图参数
bar_width = 0.25
bar_positions = np.arange(num_factors)

# 绘制柱状图
for i in range(num_portfolios):
    ax.bar(bar_positions + i * bar_width, exposures[:, i], width=bar_width, label=f"组合{i + 1}")

# 设置坐标轴
ax.set_xlabel("风格因子")
ax.set_ylabel("敞口")
ax.set_xticks(bar_positions + bar_width)
ax.set_xticklabels(style_factors)
ax.set_ylim(-3, 1)

# 设置标题
plt.title("3种组合风格因子敞口比较")

# 添加图例
ax.legend(loc='upper right')

# 显示图像
plt.show()

