from sklearn.metrics import mean_squared_error
from data3 import get_data, get_dataloader
import numpy as np
import matplotlib.pyplot as plt
import joblib
import torch
import math
from skorch import NeuralNetRegressor, NeuralNet
from model import get_1dcnn_model, RegressionCNN
import random

x_train, x_val, x_test, y_train, y_val, y_test = get_data("pfas_nalv - 副本.xlsx", random=9204)
ox_train = x_train
oy_train = y_train

cnn_gbr = get_1dcnn_model(7)
net = RegressionCNN()

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)
y_val = np.ravel(y_val)
print(type(ox_train))
# 选择的元素数量
num_elements = 34
random.seed(9402)
# 随机选择相同位置的元素
selected_indices = random.sample(range(len(ox_train)), num_elements)

ox_train = ox_train[selected_indices]
oy_train = oy_train[selected_indices]


plt.rcParams['font.family'] = 'serif'

plt.rcParams['font.serif'] = 'Times new Roman'

plt.rcParams['font.size'] = 12

cnn_gbr.load_state_dict(torch.load('checkpoint/1DGBCNN_model.pth'), False)
cnn_gbr.eval()

net.load_state_dict(torch.load('checkpoint/1DCNN_model.pth'), False)
net.eval()

y_pred = net(x_test)
y_pred = y_pred.detach().numpy()

y_true = y_test
y_pred2 = cnn_gbr(x_test).detach().numpy()

oy_train_pre = net(ox_train)
oy_train_pre = oy_train_pre.detach().numpy()
oy_train_pre2 = cnn_gbr(ox_train).detach().numpy()
# oy_train_pre2 = oy_train_pre2.detach().numpy()
plt.figure(figsize=(8.3, 3.5))
plt.rcParams['figure.dpi'] = 200
plt.subplot(121)
plt.scatter(y_true, y_pred, marker='.', color='red', label='Test')
plt.scatter(oy_train, oy_train_pre, marker='.', color='blue', label='Train')
plt.plot(y_true, y_true, 'g-', lw=1)  # 画出 y=x 的虚线

# 在图表中添加均方误差和均方根误差的标注
# plt.annotate("test_MSE: {:.2f}".format(mse), xy=(0.1, 0.9), xycoords='axes fraction')
# plt.annotate("test_RMSE: {:.2f}".format(rmse), xy=(0.1, 0.85), xycoords='axes fraction')

# 添加坐标轴标签和标题
plt.xlabel('Actual rejection', fontsize=12)
plt.ylabel('Predicted rejection', fontsize=12)
plt.title('1D-CNN')
plt.legend()

# 绘制预测结果与真实值之间的散点图
plt.subplot(122)
plt.scatter(y_true, y_pred2, marker='.', color='red', label='Test')
plt.scatter(oy_train, oy_train_pre2, marker='.', color='blue', label='Train')
plt.plot(y_true, y_true, 'g-', lw=1)  # 画出 y=x 的虚线

# 在图表中添加均方误差和均方根误差的标注
# 计算均方误差和均方根误差
mse = mean_squared_error(y_test, y_pred2)
rmse = np.sqrt(mse)
# plt.annotate("test_MSE: {:.2f}".format(mse), xy=(0.1, 0.9), xycoords='axes fraction')
# plt.annotate("test_RMSE: {:.2f}".format(rmse), xy=(0.1, 0.85), xycoords='axes fraction')

# 添加坐标轴标签和标题
plt.xlabel('Actual rejection', fontsize=12)
plt.ylabel('Predicted rejection', fontsize=12)
plt.title('1D-GBCNN')
plt.legend()
plt.tight_layout()
# 显示图表
plt.show()

# plt.savefig('Figure_2.tif', format='tif')
