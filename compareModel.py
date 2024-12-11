from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

plt.rcParams['font.family'] = 'serif'

plt.rcParams['font.serif'] = 'Times new Roman'

plt.rcParams['font.size'] = 12

# plt.rcParams['figure.figsize']=10,6
R2 = [0.822, 0.742, 0.902, 0.753, 0.846, 0.679, 0.868]
RMSE = [3.199, 3.854, 2.379, 3.766, 2.971, 4.296, 2.758]
models = ['1D-CNN', 'RF', '1D-GBCNN', 'SVR', 'XGB', 'AdaBoost', 'GBDT']
color = [(235/255, 181/255, 109/255), (69/255, 117/255, 177/255),(244/255, 127/255, 116/255),
         (153/255, 203/255, 111/255), (152/255, 216/255, 59/255), (47/255, 105/255, 142/255),
         (239/255, 144/255, 42/255)]  # 颜色设置

# plt.axhline(min(R2), linestyle=':')
# ax = sns.barplot(x=models, y=R2)
# # 创建柱状图
# plt.bar(models, R2, color=color)
# # 获取当前 y 轴的上限
# current_ylim = ax.get_ylim()
# # 计算新的 y 轴上限，增加 10%
# new_ylim = (current_ylim[0], current_ylim[1] * 1.05)
# # 设置新的 y 轴范围
# ax.set_ylim(new_ylim)
#
# # 添加标题和标签
# # plt.title('Bar Chart Example')
# plt.xlabel('Model', fontsize=12)
# plt.ylabel('R²', fontsize=12)
#
# for p in ax.patches:
#     width, height = p.get_width(), p.get_height()
#     x, y = p.get_xy()
#     ax.annotate(f'{height}', (x + width / 2, y + height * 1.02), ha='center', fontsize=10, color='black')
# # 显示图形
# plt.show()

plt.figure(figsize=(8.3, 3.5))
plt.subplot(121)
# plt.subplots_adjust(left=0.125, bottom=0.13, right=0.9, top=0.88, hspace=0.2, wspace=0.2)
plt.axhline(min(R2), linestyle=':')
ax = sns.barplot(x=models, y=R2)
plt.bar(models, R2, color=color)
current_ylim = ax.get_ylim()
# 计算新的 y 轴上限，增加 10%
new_ylim = (current_ylim[0], current_ylim[1] * 1.05)
# 设置新的 y 轴范围
ax.set_ylim(new_ylim)

# 添加标题和标签
# plt.title('Bar Chart Example')
plt.xlabel('Model', fontsize=12)
plt.ylabel('R²', fontsize=12)
plt.xticks(fontsize=8)
# plt.text('1D-CNN', 1.2, 'A', fontsize=12)
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height}', (x + width / 2, y + height * 1.02), ha='center', fontsize=8, color='black')


plt.subplot(122)
# plt.subplots_adjust(left=0.125, bottom=0.13, right=0.9, top=0.88, hspace=0.2, wspace=0.2)
plt.axhline(min(RMSE), linestyle=':')
ax = sns.barplot(x=models, y=RMSE)
plt.bar(models, RMSE, color=color)
current_ylim = ax.get_ylim()
# 计算新的 y 轴上限，增加 10%
new_ylim = (current_ylim[0], current_ylim[1] * 1.05)
# 设置新的 y 轴范围
ax.set_ylim(new_ylim)

# 添加标题和标签
# plt.title('Bar Chart Example')
plt.xlabel('Model', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.xticks(fontsize=8)
# plt.text('1D-CNN', 4.5, 'B', fontsize=12)
for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy()
    ax.annotate(f'{height}', (x + width / 2, y + height * 1.02), ha='center', fontsize=8, color='black')
# 显示图表


plt.tight_layout()
plt.show()
# top=0.88,
# bottom=0.13,
# left=0.125,
# right=0.9,
# hspace=0.2,
# wspace=0.2
# plt.savefig('Figure_3.tif', format='tif')