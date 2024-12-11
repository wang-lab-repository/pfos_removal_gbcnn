from sklearn.metrics import mean_squared_error, make_scorer, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from data3 import get_data, get_dataloader
from sklearn.model_selection import GridSearchCV
import numpy as np
import torch
import math
from skorch import NeuralNetRegressor, NeuralNet
from model import get_model, get_1dcnn_model

# from featurecaptum.attr import IntegratedGradients
x_train, x_val, x_test, y_train, y_val, y_test = get_data("pfas_nalv - 副本.xlsx", random=9204, percent=1)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)
y_val = np.ravel(y_val)
print(x_train)
# 创建梯度增强回归模型
# 定义参数网格
param_grid = {
    'n_estimators': [80, 100, 120, 130, 150],
    'learning_rate': [0.1, 0.05, 0.01, 0.2],
    'max_depth': [3, 4, 5, 6],
    'alpha': [0.5, 0.7, 0.9],
    'subsample': [1]
}

# 初始化模型
mymodel = get_1dcnn_model()

# 加载模型参数
model_path = 'checkpoint/new_model.pth'
mymodel.load_state_dict(torch.load(model_path))

# 设置模型为评估模式
mymodel.eval()


class PytorchNet(NeuralNetRegressor):
    def predict(self, X):
        return self.predict_proba(X)


nnr = PytorchNet(mymodel, criterion=torch.nn.MSELoss, warm_start=True)

import warnings

warnings.filterwarnings("ignore", message="Using a target size")
# model.fit(x_train, y_train)
# 0.5数据 {'alpha': 0.5, 'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 130, 'subsample': 1}
# 0.75数据 {'alpha': 0.9, 'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 150, 'subsample': 1}
# 使用网格搜索进行参数调优 # Best Parameters: {'alpha': 0.9, 'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 130, 'subsample': 1}
# model1 = GradientBoostingRegressor(init=nnr, verbose=0)
# grid_search = GridSearchCV(model1, param_grid, cv=5)
grid_search = GradientBoostingRegressor(init=nnr, alpha=0.7, learning_rate=0.1, max_depth=6, n_estimators=70,
                                        subsample=1, verbose=0, random_state=92)

grid_search.fit(x_train, y_train)
# 输出最佳参数组合和对应的模型性能
# print("Best Parameters:", grid_search.best_params_)
# print("Best Score:", grid_search.best_score_)

# 保存模型
# joblib.dump(grid_search, '1D-CNNGB_model.pkl')

# explainer = shap.DeepExplainer(grid_search, x_train)
# # 计算训练集上的 SHAP 值
# # x_train = x_train.numpy()
# shap_values = explainer.shap_values(x_train)
# # 绘制特征重要性图表
# shap.summary_plot(shap_values, x_train)
# plt.savefig('result/gbr-nn-shaping_3-21.tif')

# # 输出最佳参数组合和对应的模型性能
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
print("------------------------1DCNN-GBR------------------------")
print(f'train: R2：{R2_train.detach().numpy()}\n')
print(f'val: R2：{R2_val.detach().numpy()}\n')
print(f'test: R2：{R2_test.detach().numpy()}\n')
print(f'train: RMSE：{np.sqrt(mean_squared_error(y_train, prediction_train))}\n')
print(f'val: RMSE：{np.sqrt(mean_squared_error(y_val, prediction_val))}\n')
print(f'test: RMSE：{np.sqrt(mean_squared_error(y_test, prediction_test))}\n')
