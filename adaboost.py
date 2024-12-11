from sklearn.metrics import mean_squared_error, make_scorer, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from data3 import get_data, get_dataloader
import numpy as np

import torch
import math
from sklearn.ensemble import AdaBoostRegressor

x_train, x_val, x_test, y_train, y_val, y_test = get_data("pfas_nalv - 副本.xlsx", random=9204)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)
y_val = np.ravel(y_val)
# print(x_train)
param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'learning_rate': [0.01, 0.1, 1.0],
    'loss': ['linear', 'square', 'exponential']
}

model = AdaBoostRegressor()
# adaboost_reg = GridSearchCV(model, param_grid, cv=5)
# grid_search.fit(x_train, y_train)
# print("Best parameters found: ", adaboost_reg.best_params_)
# R2：0.6789 RMSE：4.2964 MAE：3.1418 MAPE：3.3881
adaboost_reg = AdaBoostRegressor(n_estimators=500, learning_rate=0.1, random_state=9204)

# 在训练集上训练模型
adaboost_reg.fit(x_train, y_train)

# 在验证集上评估最佳模型性能
val_predictions = adaboost_reg.predict(x_val)
val_mse = mean_squared_error(y_val, val_predictions)
print("Validation RMSE:", math.sqrt(val_mse))

# 在测试集上评估最佳模型性能
test_predictions = adaboost_reg.predict(x_test)
test_mse = mean_squared_error(y_test, test_predictions)
print("Test RMSE:", math.sqrt(test_mse))

prediction_train = adaboost_reg.predict(x_train)
prediction_val = adaboost_reg.predict(x_val)
prediction_test = adaboost_reg.predict(x_test)

y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()
y_val = torch.from_numpy(y_val).float()
R2_train = 1 - torch.mean((y_train - prediction_train) ** 2) / torch.mean((y_train - torch.mean(y_train)) ** 2)
R2_val = 1 - torch.mean((y_val - prediction_val) ** 2) / torch.mean((y_val - torch.mean(y_val)) ** 2)
R2_test = 1 - torch.mean((y_test - prediction_test) ** 2) / torch.mean((y_test - torch.mean(y_test)) ** 2)
print("------------------------AdaB------------------------")
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