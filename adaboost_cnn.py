import math
import warnings
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import _num_samples
from torch.nn import Linear
from data2 import get_data, get_dataloader


class WeakLearner(nn.Module):
    def __init__(self):
        super(WeakLearner, self).__init__()
        f1 = 25
        f2 = 27
        c1 = 20
        c2 = 20
        dropout_rate = 0.10
        t1 = 1
        self.t1 = t1
        self.conv1 = nn.Conv1d(1, c1, 4)
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(c1, c2, 4)
        self.relu = nn.ReLU()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(1 * c2, f1)
        self.fc2 = nn.Linear(f1, f2)
        self.fc3 = nn.Linear(f2, f2)
        self.fc4 = nn.Linear(f2, 1)
        self.drop = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = x.reshape(-1, 1, 13)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.fc2(x)
        temp = self.relu(x)
        x = self.relu(self.fc3(temp))
        if self.t1 == 1:
            x = self.relu(self.fc4(x + temp))
        else:
            x = self.relu(self.fc4(x))

        return x

    def predict(self, x):
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions


class FCN(nn.Module):
    def __init__(self):
        """

        :rtype: object
        """
        super().__init__()
        self.inline = Linear(13, 64)
        self.hideline1 = Linear(64, 128)
        self.hideline2 = Linear(128, 64)
        self.hideline3 = Linear(64, 8)
        self.outline = Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.inline(x)
        x = self.relu(x)
        x = self.hideline1(x)
        x = self.relu(x)
        x = self.hideline2(x)
        x = self.relu(x)
        x = self.hideline3(x)
        x = self.relu(x)
        x = self.outline(x)
        return x

    def predict(self, x):
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions


def train_weak_learner(model, data_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for inputs, labels, weights in data_loader:
            outputs = model(inputs)
            loss = torch.mean(weights * criterion(outputs, labels))
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()


# from hu_NN import FCN

x_train, x_val, x_test, y_train, y_val, y_test = get_data("pfas_nalv.xlsx", random=9204, percent=1)


def adaboost_pytorch(X, y, num_classifiers):
    N, D = X.shape
    weights = torch.ones(N) / N  # 初始化样本权重： 1/训练样本数D
    weak_learners = []  # 每一轮保存下来的模型
    alpha_values = []  # 每一个模型的权重
    torch.manual_seed(9204)
    for t in range(num_classifiers):
        # 1、在样本分布Dist1的基础上，在训练集上训练弱分类器ht
        weak_learner = WeakLearner()
        # weak_learner = FCN()
        # weak_learner.load_state_dict(torch.load('checkpoint/save_model.ckpt'), False)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(weak_learner.parameters(), lr=0.0020)

        train_data = torch.utils.data.TensorDataset(x_train, y_train, weights)
        train_load = torch.utils.data.DataLoader(
            dataset=train_data,

            batch_size=256,
            shuffle=True,

        )
        optimizer.zero_grad()
        # 启用异常检测
        torch.autograd.set_detect_anomaly(True)
        train_weak_learner(weak_learner, train_load, criterion, optimizer, num_epochs=30)
        weak_learners.append(weak_learner)
        # 2、计算分类器ht在训练集上的最大误差Et
        global_max_loss = 0
        for input, target in zip(X, y):
            # 使用模型对输入进行预测
            output = weak_learner(input)
            # 计算预测误差（这里使用均方误差作为示例）
            loss = criterion(output, target)
            # 计算当前批次的最大误差
            max_loss = torch.max(loss)
            # 更新全局最大误差
            global_max_loss = torch.max(torch.tensor(global_max_loss), max_loss)
            Et = global_max_loss.item()
        print(t + 1, "轮最大误差：", Et)
        # 3、根据求得的ht的最大误差Et，计算ht对每个样本的相对误差eti(平方误差)
        relative_errors = []
        # 当前弱分类器ht的误差率et
        et = 0
        for input, target, weight in zip(X, y, weights):
            # 使用模型对输入进行预测
            output = weak_learner(input)

            # 计算预测误差（这里使用均方误差作为示例）
            loss = criterion(output, target)

            # 计算每个样本的相对误差（平方误差）
            max_error = Et
            # 遍历每一个样本损失以及对应的数据样本权重

            epsilon = 1e-8
            # relative_error = (loss / torch.max(max_error, torch.tensor(epsilon))) ** 2  # 每个样本相对误差eti
            # relative_error = ((target - output) / max_error) ** 2
            relative_error = torch.sqrt(loss) / (max_error)

            # relative_error = loss / (max_error ** 2)
            # 将每个样本的相对误差加入列表中
            relative_errors.append(relative_error)
            # 4、根据上一步求得得样本相对误差eti，计算出当前弱分类器ht的误差率et，即数据集中所有样本的权重与误差之乘积的和
            et = et + relative_error * weight
        print("relative_errors:", relative_errors[:10])
        # 5、更新当前弱分类器ht的权重 wt
        wt = et / (1 - et)
        alpha_values.append(wt)
        # 6、更新数据样本的权重分布
        Zt = 0
        for i in range(N):
            Zt = Zt + weights[i] * (wt ** (1 - relative_errors[i]))
        print("Zt:", Zt)
        print("更新前weights:", weights[:10])
        weights_copy = weights.clone()
        for i in range(N):
            weights_copy[i] = (weights[i] / Zt) * (wt ** (1 - relative_errors[i]))
            # weights[i] = (weights[i] / Zt) * (wt**(1-relative_errors[i]))
        print("weights_copy:", weights_copy[:10])
        weights = weights_copy
        print("更新后weights:", weights[:10])
        print("weak_learners:", len(weak_learners))
        print("alpha_values:", alpha_values)

    return weak_learners, alpha_values


def stable_cumsum(arr, axis=None, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum.

    Warns if the final cumulative sum does not match the sum (up to the chosen
    tolerance).

    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat.
    axis : int, default=None
        Axis along which the cumulative sum is computed.
        The default (None) is to compute the cumsum over the flattened array.
    rtol : float, default=1e-05
        Relative tolerance, see ``np.allclose``.
    atol : float, default=1e-08
        Absolute tolerance, see ``np.allclose``.

    Returns
    -------
    out : ndarray
        Array with the cumulative sums along the chosen axis.
    """
    out = np.cumsum(arr, axis=axis, dtype=np.float64)
    expected = np.sum(arr, axis=axis, dtype=np.float64)
    if not np.all(
            np.isclose(
                out.take(-1, axis=axis), expected, rtol=rtol, atol=atol, equal_nan=True
            )
    ):
        warnings.warn(
            (
                "cumsum was found to be unstable: "
                "its last element does not correspond to sum"
            ),
            RuntimeWarning,
        )
    return out


def _get_median_predict(weak_learners, alpha_values, X, limit):
    # Evaluate predictions of all estimators
    predictions = np.array([(est.predict(X).squeeze()).numpy() for est in weak_learners[:limit]]).T

    # print("predictions shape:", predictions.shape)
    # print("predictions type:", type(predictions))
    # print("predictions:", predictions)
    # Sort the predictions
    sorted_idx = np.argsort(predictions, axis=1)
    # print("sorter_id:", sorted_idx)
    sorted_idx = sorted_idx.astype(int)
    # print("sorter_id:", sorted_idx)
    # Find index of median prediction for each sample
    # print("alpha_values shape:", alpha_values.shape)
    # print("alpha_values:", alpha_values)
    # print("alpha_values[sorted_idx] shape:", alpha_values[sorted_idx].shape)
    # print("alpha_values[sorted_idx]:", alpha_values[sorted_idx])
    weight_cdf = stable_cumsum(alpha_values[sorted_idx], axis=1)
    median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1][:, np.newaxis]
    median_idx = median_or_above.argmax(axis=1)

    median_estimators = sorted_idx[np.arange(_num_samples(X)), median_idx]

    # Return median predictions
    return predictions[np.arange(_num_samples(X)), median_estimators]


nums_epoch = 4
weak_learners, alpha_values = adaboost_pytorch(x_train, y_train, nums_epoch)
alpha_values = [tensor.detach().numpy() for tensor in alpha_values]
alpha_values = np.array(alpha_values)
y_pred = _get_median_predict(weak_learners, alpha_values, torch.Tensor(x_test), nums_epoch)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

prediction_train = _get_median_predict(weak_learners, alpha_values, torch.Tensor(x_train), nums_epoch)
prediction_val = _get_median_predict(weak_learners, alpha_values, torch.Tensor(x_val), nums_epoch)
prediction_test = _get_median_predict(weak_learners, alpha_values, torch.Tensor(x_test), nums_epoch)

prediction_train = np.array(prediction_train)
prediction_val = np.array(prediction_val)
prediction_test = np.array(prediction_test)

print("prediction_test:", prediction_test)
print("y_test:", y_test)

print("------------------------adaboost-pytorch------------------------")
print(f'train: R2：{r2_score(y_train, prediction_train)}\n')
print(f'val: R2：{r2_score(y_val, prediction_val)}\n')
print(f'test: R2：{r2_score(y_test, prediction_test)}\n')
print(f'train: RMSE：{np.sqrt(mean_squared_error(y_train, prediction_train))}\n')
print(f'val: RMSE：{np.sqrt(mean_squared_error(y_val, prediction_val))}\n')
print(f'test: RMSE：{np.sqrt(mean_squared_error(y_test, prediction_test))}\n')
