from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['font.family'] = 'serif'

plt.rcParams['font.serif'] = 'Times new Roman'

plt.rcParams['font.size'] = 13

# plt.rcParams['figure.figsize']=10,6
R2 = [0.822, 0.742, 0.902, 0.753, 0.846, 0.679, 0.868]
RMSE = [3.199, 3.854, 2.379, 3.766, 2.971, 4.296, 2.758]
models = ['1D-CNN', 'RF', '1D-GBCNN', 'SVR', 'XGB', 'AdaBoost', 'GBDT']
color = [(235/255, 181/255, 109/255), (69/255, 117/255, 177/255),(244/255, 127/255, 116/255),
         (153/255, 203/255, 111/255), (152/255, 216/255, 59/255), (47/255, 105/255, 142/255),
         (239/255, 144/255, 42/255)]  # 颜色设置

plt.axhline(min(RMSE), linestyle=':')
ax = sns.barplot(x=models, y=RMSE)
# 创建柱状图
plt.bar(models, RMSE, color=color)
# 获取当前 y 轴的上限
current_ylim = ax.get_ylim()
# 计算新的 y 轴上限，增加 10%
new_ylim = (current_ylim[0], current_ylim[1] * 1.05)
# 设置新的 y 轴范围
ax.set_ylim(new_ylim)

# 添加标题和标签
# plt.title('Bar Chart Example')
plt.xlabel('Model')
plt.ylabel('RMSE')

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height}', (x + width / 2, y + height * 1.02), ha='center', fontsize=10, color='black')
# 显示图形
plt.show()
