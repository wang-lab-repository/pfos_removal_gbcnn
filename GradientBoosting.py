from captum.attr import ShapleyValueSampling
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from data3 import get_data, get_dataloader
import numpy as np
import matplotlib.pyplot as plt
import torch
import math

from model import get_model
import shap
x_train, x_val, x_test, y_train, y_val, y_test = get_data("pfas_nalv - 副本.xlsx", random=9204, percent=1)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)
y_val = np.ravel(y_val)


print("过采样---------data_sampled", x_train)

# 创建梯度增强回归模型
# 定义参数网格
param_grid = {
    'n_estimators': [50, 60, 70, 80, 100],
    'learning_rate': [0.1, 0.05, 0.01, 0.15],
    'max_depth': [3, 4, 5, 6],
    'alpha': [0.7, 0.8, 0.9],
    'subsample': [0.9, 1]
}

# 创建梯度增强回归模型
model = GradientBoostingRegressor()

# 使用网格搜索进行参数调优
grid_search = GridSearchCV(model, param_grid, cv=5)

# Best Parameters: {'alpha': 0.7, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 80, 'subsample': 1}test r2 0.85

# learning_rate=0.1, max_depth=5, n_estimators=80, alpha=0.7, random_state=92
# R2：0.8677 RMSE：2.7580 MAE：1.6636 MAPE：1.8446
grid_search = GradientBoostingRegressor(learning_rate=0.1, max_depth=5, n_estimators=80, alpha=0.7, random_state=92)

grid_search.fit(x_train, y_train)

# 输出最佳参数组合和对应的模型性能
# print("Best Parameters:", grid_search.best_params_)
# print("Best Score:", grid_search.best_score_)


# 在验证集上评估最佳模型性能
val_predictions = grid_search.predict(x_val)
val_mse = mean_squared_error(y_val, val_predictions)
print("Validation RMSE:", math.sqrt(val_mse))

# 在测试集上评估最佳模型性能
test_predictions = grid_search.predict(x_test)
test_mse = mean_squared_error(y_test, test_predictions)
print("Test RMSE:", math.sqrt(test_mse))


prediction_train = grid_search.predict(x_train)
prediction_val = grid_search.predict(x_val)
prediction_test = grid_search.predict(x_test)


y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()
y_val = torch.from_numpy(y_val).float()
R2_train = 1 - torch.mean((y_train - prediction_train) ** 2) / torch.mean((y_train - torch.mean(y_train)) ** 2)
R2_val = 1 - torch.mean((y_val - prediction_val) ** 2) / torch.mean((y_val - torch.mean(y_val)) ** 2)
R2_test = 1 - torch.mean((y_test - prediction_test) ** 2) / torch.mean((y_test - torch.mean(y_test)) ** 2)
print("------------------------GBDT------------------------")
print(f'train: R2：{R2_train.detach().numpy()}\n')
print(f'val: R2：{R2_val.detach().numpy()}\n')
print(f'test: R2：{R2_test.detach().numpy()}\n')
print(f'train: RMSE：{np.sqrt(mean_squared_error(y_train, prediction_train))}\n')
print(f'val: RMSE：{np.sqrt(mean_squared_error(y_val, prediction_val))}\n')
print(f'test: RMSE：{np.sqrt(mean_squared_error(y_test, prediction_test))}\n')


def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error.
    Note: It assumes that y_true does not contain zeros to avoid division by zero.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_indices = y_true != 0  # Avoid division by zero
    y_true = y_true[non_zero_indices]
    y_pred = y_pred[non_zero_indices]

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# 计算训练集、验证集和测试集上的MAE
mae_train = mean_absolute_error(y_train.numpy(), prediction_train)
mae_val = mean_absolute_error(y_val.numpy(), prediction_val)
mae_test = mean_absolute_error(y_test.numpy(), prediction_test)

# 计算训练集、验证集和测试集上的MAPE
mape_train = mean_absolute_percentage_error(y_train.numpy(), prediction_train)
mape_val = mean_absolute_percentage_error(y_val.numpy(), prediction_val)
mape_test = mean_absolute_percentage_error(y_test.numpy(), prediction_test)

print(f'train: MAE：{mae_train}\n')
print(f'val: MAE：{mae_val}\n')
print(f'test: MAE：{mae_test}\n')
print(f'train: MAPE：{mape_train}\n')
print(f'val: MAPE：{mape_val}\n')
print(f'test: MAPE：{mape_test}\n')