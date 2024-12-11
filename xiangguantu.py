import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path = 'pfas_nalv - 副本.xlsx'
columns_list = ['Membrane type', 'MWCO(Da)', 'Pore size(nm)', 'water flux(LMH)', 'Temperature (˚C)', 'PFOS con (ppb)',
                'pH',
                'Pressure (MPa)', 'Divalent cations (mmol/L)',
                'Monovalent cations (mmol/L)', 'Trivalent cations (mmol/L)',
                'PFOS rejection (%)']
df_all = pd.read_excel(path)
df_all = df_all.drop(['Data'], axis=1)

# 数据分布图
# sns.distplot(df_all['MWCO(Da)'])

# 不同膜的截留率数据 箱线图
sns.boxplot(x='Membrane type', y='PFOS rejection (%)', data=df_all)

# 计算相关性矩阵
corr_matrix = df_all.corr()
# 绘制相关性热力图
plt.figure(figsize=(10, 8))  # 设置图形大小
plt.rcParams['font.family'] = 'serif'

plt.rcParams['font.serif'] = 'Times new Roman'

plt.rcParams['font.size'] = 13
ax = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f',
                 annot_kws={'size': 10},
                 cbar_kws={'shrink': .8},
                 center=0)


# 设置横坐标标签倾斜45度
plt.xticks(rotation=45, ha='right')  # ha='right' 是为了让标签不与热力图中的值重叠

# 调整图形边距（如果需要的话，可以在这里再次调用 plt.tight_layout()）
plt.tight_layout()

# 显示图形
plt.show()
