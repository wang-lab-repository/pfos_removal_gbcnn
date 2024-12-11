from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, make_scorer, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from data import get_data, get_dataloader
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from sklearn.ensemble import VotingRegressor
from model import get_model
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer

x_train, x_val, x_test, y_train, y_val, y_test = get_data("pfas_nalv.xlsx", random=9204)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)
y_val = np.ravel(y_val)
print(x_train)
# 创建梯度增强回归模型
# 定义参数网格
param_grid = {
    'gbr__n_estimators': [100, 130, 150],
    'gbr__learning_rate': [0.1, 0.05, 0.01, 0.2],
    'gbr__max_depth': [3, 4, 5],
    'gbr__alpha': [0.8, 0.9],
    'gbr__subsample': [1]
}


class PyTorchModelWrapper:
    def __init__(self, model_path):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        model = get_model()  # 实例化你的PyTorch模型类
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def predict(self, input_data):
        with torch.no_grad():
            input_tensor = torch.Tensor(input_data)  # 将输入数据转换为张量
            output = self.model(input_tensor)
            return output.numpy()  # 返回预测结果


# net = get_model()
# net.load_state_dict(torch.load('checkpoint/save_model.ckpt', map_location=torch.device('cpu')))

from sklearn.base import BaseEstimator, RegressorMixin


class PyTorchEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, pytorch_model):
        self.pytorch_model = pytorch_model

    def fit(self, X, y):
        # 可选：根据需要实现拟合过程
        pass

    def predict(self, X):
        return self.pytorch_model.predict(X)

random_state = 4  #
# 创建梯度增强回归模型
# 使用训练好的神经网络模型作元学习器
pytorch_estimator = PyTorchEstimator(PyTorchModelWrapper('checkpoint/save_model.ckpt'))
# 定义其他基本模型
base_models = [
    ('rf', RandomForestRegressor(max_depth=10, max_features=3, min_samples_split=2, n_estimators=50, random_state=random_state)),
    ('gb', GradientBoostingRegressor(alpha=0.9, learning_rate=0.05, max_depth=5, n_estimators=100, subsample=1, random_state=random_state)),
    ('nn', pytorch_estimator)
]
meta_model = RandomForestRegressor(random_state = random_state)
# 创建堆叠回归器
# stacked_model = StackingRegressor(estimators=base_models , final_estimator=LinearRegression())
stacked_model = StackingCVRegressor(regressors=[model for name, model in base_models],
                                    meta_regressor=meta_model,
                                    cv=KFold(n_splits=5, shuffle=True, random_state=random_state))


# 定义超参数搜索空间
param_space = {
    'meta_regressor__n_estimators': Integer(50, 100),  # 72, 4, 4
    'meta_regressor__max_depth': Integer(3, 10),
    'meta_regressor__min_samples_split': Integer(2, 10)
}

# 使用贝叶斯优化进行参数搜索
bayes_search = BayesSearchCV(stacked_model, param_space, cv=KFold(n_splits=5, shuffle=True, random_state=42))
bayes_search.fit(x_train, y_train)
# 输出最佳参数组合和对应的得分
print("Best Parameters: ", bayes_search.best_params_)
print("Best Score: {:.2f}".format(bayes_search.best_score_))

# 预测并评估模型
y_pred = bayes_search.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# 在验证集上评估最佳模型性能
val_predictions = bayes_search.predict(x_val)
val_mse = mean_squared_error(y_val, val_predictions)
print("Validation RMSE:", math.sqrt(val_mse))

# 在测试集上评估最佳模型性能
test_predictions = bayes_search.predict(x_test)
test_mse = mean_squared_error(y_test, test_predictions)
print("Test RMSE:", math.sqrt(test_mse))

prediction_train = bayes_search.predict(x_train)
prediction_val = bayes_search.predict(x_val)
prediction_test = bayes_search.predict(x_test)

y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()
y_val = torch.from_numpy(y_val).float()
R2_train = 1 - torch.mean((y_train - prediction_train) ** 2) / torch.mean((y_train - torch.mean(y_train)) ** 2)
R2_val = 1 - torch.mean((y_val - prediction_val) ** 2) / torch.mean((y_val - torch.mean(y_val)) ** 2)
R2_test = 1 - torch.mean((y_test - prediction_test) ** 2) / torch.mean((y_test - torch.mean(y_test)) ** 2)
print("------------------------结果------------------------")
print(f'train: R2：{R2_train.detach().numpy()}\n')
print(f'val: R2：{R2_val.detach().numpy()}\n')
print(f'test: R2：{R2_test.detach().numpy()}\n')
print(f'train: RMSE：{np.sqrt(mean_squared_error(y_train, prediction_train))}\n')
print(f'val: RMSE：{np.sqrt(mean_squared_error(y_val, prediction_val))}\n')
print(f'test: RMSE：{np.sqrt(mean_squared_error(y_test, prediction_test))}\n')



#----------------------------------------------------------------
# # 训练堆叠回归器
# stacked_model.fit(x_train, y_train)
# # 预测并评估模型
# y_pred = stacked_model.predict(x_test)
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')
#
# # 在验证集上评估最佳模型性能
# val_predictions = stacked_model.predict(x_val)
# val_mse = mean_squared_error(y_val, val_predictions)
# print("Validation RMSE:", math.sqrt(val_mse))
#
# # 在测试集上评估最佳模型性能
# test_predictions = stacked_model.predict(x_test)
# test_mse = mean_squared_error(y_test, test_predictions)
# print("Test RMSE:", math.sqrt(test_mse))
#
# prediction_train = stacked_model.predict(x_train)
# prediction_val = stacked_model.predict(x_val)
# prediction_test = stacked_model.predict(x_test)
#
# y_train = torch.from_numpy(y_train).float()
# y_test = torch.from_numpy(y_test).float()
# y_val = torch.from_numpy(y_val).float()
# R2_train = 1 - torch.mean((y_train - prediction_train) ** 2) / torch.mean((y_train - torch.mean(y_train)) ** 2)
# R2_val = 1 - torch.mean((y_val - prediction_val) ** 2) / torch.mean((y_val - torch.mean(y_val)) ** 2)
# R2_test = 1 - torch.mean((y_test - prediction_test) ** 2) / torch.mean((y_test - torch.mean(y_test)) ** 2)
# print("------------------------结果------------------------")
# print(f'train: R2：{R2_train.detach().numpy()}\n')
# print(f'val: R2：{R2_val.detach().numpy()}\n')
# print(f'test: R2：{R2_test.detach().numpy()}\n')
# print(f'train: RMSE：{np.sqrt(mean_squared_error(y_train, prediction_train))}\n')
# print(f'val: RMSE：{np.sqrt(mean_squared_error(y_val, prediction_val))}\n')
# print(f'test: RMSE：{np.sqrt(mean_squared_error(y_test, prediction_test))}\n')

#
# #
# best_model = grid_search.best_estimator_
# # 创建 SHAP 解释器
# features = ['Temperature (˚C)', 'PFOS con (ppb)', 'pH',
#                     'Pressure (MPa)', 'Divalent cations (mmol/L)',
#                     'Monovalent cations (mmol/L)', 'Trivalent cations (mmol/L)',
#             'Membrane type__ESNA1-K1', 'Membrane type__HYDRACORE', 'Membrane type__NE70',
#             'Membrane type__NF270', 'Membrane type__PMIA', 'Membrane type__Poly(piperazineamide) NF']
# explainer = shap.DeepExplainer(best_model)
# # 计算训练集上的 SHAP 值
# x_train = x_train.numpy()
# shap_values = explainer.shap_values(x_train)
# # 绘制特征重要性图表
# shap.summary_plot(shap_values, x_train, max_display=8)
# plt.show()
