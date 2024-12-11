import torch
import torch.nn as nn

from early_stopping import EarlyStopping
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from data3 import get_data, get_dataloader
from sklearn.ensemble import AdaBoostRegressor
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load
import math
import matplotlib.pyplot as plt
from model import get_model



class GradientBoostingRegressorTorch(nn.Module):
    def __init__(self, n_estimators=10, learning_rate=0.005):
        super(GradientBoostingRegressorTorch, self).__init__()
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = []

    def fit(self, X, y):
        model = get_model()
        model.load_state_dict(torch.load('checkpoint/save_model.ckpt'), False)
        # model.load_state_dict(torch.load('checkpoint/qnn_8_4-1.pth'))
        residuals = y.clone().detach() - model(X)
        self.models.append(model)
        # print("The shape of res is ",residuals.shape)
        for i in range(1, self.n_estimators):
            model = get_model()

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
            # 冻结部分层
            # for param in model.parameters():
            #     param.requires_grad = False
            # for param in model.outline.parameters():
            #     param.requires_grad = True
            # 训练模型
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(residuals, dtype=torch.float32).view(-1, 1)

            for epoch in range(100):
                model.train()
                optimizer.zero_grad()
                y_pred = model(X_tensor)
                # print("The shape of y_pred is ", y_pred.shape)
                # print("The shape of y_tensor is ", y_tensor.shape)
                loss = criterion(y_pred, y_tensor)
                loss.backward()
                optimizer.step()

            self.models.append(model)
            model.eval()
            with torch.no_grad():
                predictions = model(X_tensor)

                residuals -= predictions

    def predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_pred = np.zeros(X.shape[0])
        for model in self.models:
            model.eval()
            with torch.no_grad():
                y_pred += model(X_tensor).numpy().flatten()
        return y_pred

    def save(self):
        torch.save(self.state_dict(), 'checkpoint/gbr_torch_nn_12-3.pth')
    # def predict(self, X)
    #     X_tensor = torch.tensor(X, dtype=torch.float32)
    #     y_pred = np.zeros(X.shape[0])
    #     for i in range(len(self.models)):
    #         model = self.models[i]
    #         model.eval()
    #         with torch.no_grad():
    #             if i == 0:
    #                 y_pred += model(X_tensor).numpy().flatten()
    #             else:
    #                 y_pred += self.learning_rate * model(X_tensor).numpy().flatten()
    #     return y_pred


# 实例化并训练模型
x_train, x_val, x_test, y_train, y_val, y_test = get_data("pfas_nalv - 副本.xlsx", random=9204, percent=1)
train_load = get_dataloader(x_train, y_train)
gbr_torch = GradientBoostingRegressorTorch()
gbr_torch.fit(x_train, y_train)

# 在验证集上评估最佳模型性能
val_predictions = gbr_torch.predict(x_val)
val_mse = mean_squared_error(y_val, val_predictions)
print("Validation RMSE:", math.sqrt(val_mse))

# 在测试集上评估最佳模型性能
test_predictions = gbr_torch.predict(x_test)
test_mse = mean_squared_error(y_test, test_predictions)
print("Test RMSE:", math.sqrt(test_mse))

prediction_train = gbr_torch.predict(x_train)
prediction_val = gbr_torch.predict(x_val)
prediction_test = gbr_torch.predict(x_test)

y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
R2_train = r2_score(y_train, prediction_train)
R2_val = r2_score(y_val, prediction_val)
R2_test = r2_score(y_test, prediction_test)
print("------------------------结果------------------------")
print(f'train: R2：{R2_train}\n')
print(f'val: R2：{R2_val}\n')
print(f'test: R2：{R2_test}\n')
print(f'train: RMSE：{np.sqrt(mean_squared_error(y_train, prediction_train))}\n')
print(f'val: RMSE：{np.sqrt(mean_squared_error(y_val, prediction_val))}\n')
print(f'test: RMSE：{np.sqrt(mean_squared_error(y_test, prediction_test))}\n')
gbr_torch.save()