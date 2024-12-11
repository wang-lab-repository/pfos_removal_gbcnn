import torch
import torch.nn as nn
import torch.optim as optim
from utils import mix_seed
from data3 import get_data, get_dataloader
import torch.utils.data as Data
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from model import RegressionCNN

# # 定义一个简单的神经网络树
# class RegressionCNN(nn.Module):
#     def __init__(self):
#         super(RegressionCNN, self).__init__()
#         # 第一层卷积层
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=2)
#         self.pool1 = nn.MaxPool1d(kernel_size=2)
#         # 第二层卷积层
#         self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=2)
#         self.pool2 = nn.MaxPool1d(kernel_size=4)
#         # 全连接层
#         self.fc = nn.Linear(in_features=32, out_features=1)  # 假设我们只有一个输出值
#
#     def forward(self, x):
#         x = x.reshape(-1, 1, 16)
#         # 第一层卷积和池化
#         x = self.pool1(torch.relu(self.conv1(x)))
#         # 第二层卷积和池化
#         x = self.pool2(torch.relu(self.conv2(x)))
#         # 扁平化处理
#         x = x.view(x.size(0), -1)
#         # 全连接层
#         x = self.fc(x)  # 不使用激活函数，因为这是回归任务
#         return x
#
#
# # 定义GrowNet
# class GrowNet(nn.Module):
#     def __init__(self, num_trees=5):
#         super(GrowNet, self).__init__()
#         self.trees = nn.ModuleList([RegressionCNN() for _ in range(num_trees)])
#
#     def forward(self, x):
#         predictions = []
#         residual = x.clone()
#         for tree in self.trees:
#             prediction = tree(residual)
#             predictions.append(prediction)
#             residual = residual - prediction
#         return sum(predictions)


x_train, x_val, x_test, y_train, y_val, y_test = get_data("pfas_nalv - 副本.xlsx", random=9204, percent=1)
train_load = get_dataloader(x_train, y_train)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"训练设备：{device}")
mix_seed(9204)

val_dataset = Data.TensorDataset(torch.tensor(x_val, dtype=torch.float32).clone().detach(),
                                 torch.tensor(y_val, dtype=torch.float32).clone().detach())
test_dataset = Data.TensorDataset(torch.tensor(x_test, dtype=torch.float32).clone().detach(),
                                  torch.tensor(y_test, dtype=torch.float32).clone().detach())

train_loader = train_load
val_loader = Data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = Data.DataLoader(test_dataset, batch_size=32, shuffle=False)

num_trees = 7
model = RegressionCNN().to(device)

# 损失函数和优化器
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
train_loss_list = []
val_loss_list = []


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=10):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            # print("inputs shape", inputs.shape)
            # print("outputs shape", outputs.shape)
            # print("targets shape", targets.shape)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_loss_list.append(val_loss)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_wts = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print('Early stopping!')
            break
    model.load_state_dict(best_model_wts)
    return model


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

if __name__ == "__main__":
    # Generate some example data
    x_train, x_val, x_test, y_train, y_val, y_test = get_data("pfas_nalv - 副本.xlsx", random=9204, percent=1)
    train_load = get_dataloader(x_train, y_train)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"训练设备：{device}")
    mix_seed(9204)

    val_dataset = Data.TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    test_dataset = Data.TensorDataset(torch.tensor(x_test, dtype=torch.float32),
                                      torch.tensor(y_test, dtype=torch.float32))

    train_loader = train_load
    val_loader = Data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = Data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    # Initialize model, criterion, and optimizer
    model = RegressionCNN()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # Train the model
    trained_model = train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=1000, patience=100)

    trained_model.eval()

    prediction_train = trained_model(x_train)
    prediction_val = trained_model(x_val)
    prediction_test = trained_model(x_test)

    plt.plot(train_loss_list, label='Training loss')
    plt.plot(val_loss_list, label='Val loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Val Loss')
    plt.legend()
    plt.show()

    R2_train = 1 - torch.mean((y_train - prediction_train) ** 2) / torch.mean(
        (y_train - torch.mean(y_train)) ** 2)
    R2_val = 1 - torch.mean((y_val - prediction_val) ** 2) / torch.mean(
        (y_val - torch.mean(y_val)) ** 2)
    R2_test = 1 - torch.mean((y_test - prediction_test) ** 2) / torch.mean(
        (y_test - torch.mean(y_test)) ** 2)
    print("------------------------结果------------------------")
    print(f'train: R2：{R2_train.detach().numpy()}\n')
    print(f'val: R2：{R2_val.detach().numpy()}\n')
    print(f'test: R2：{R2_test.detach().numpy()}\n')
    print(f'train: RMSE：{np.sqrt(mean_squared_error(y_train.detach().numpy(), prediction_train.detach().numpy()))}\n')
    print(f'val: RMSE：{np.sqrt(mean_squared_error(y_val.detach().numpy(), prediction_val.detach().numpy()))}\n')
    print(f'test: RMSE：{np.sqrt(mean_squared_error(y_test.detach().numpy(), prediction_test.detach().numpy()))}\n')

    # Save the trained model (optional)
    torch.save(trained_model.state_dict(), 'checkpoint/1DCNN_model.pth')


    # 计算训练集、验证集和测试集上的MAE
    mae_train = mean_absolute_error(y_train.numpy(), prediction_train.detach().numpy())
    mae_val = mean_absolute_error(y_val.numpy(), prediction_val.detach().numpy())
    mae_test = mean_absolute_error(y_test.numpy(), prediction_test.detach().numpy())

    # 计算训练集、验证集和测试集上的MAPE
    mape_train = mean_absolute_percentage_error(y_train.numpy(), prediction_train.detach().numpy())
    mape_val = mean_absolute_percentage_error(y_val.numpy(), prediction_val.detach().numpy())
    mape_test = mean_absolute_percentage_error(y_test.numpy(), prediction_test.detach().numpy())

    print(f'train: MAE：{mae_train}\n')
    print(f'val: MAE：{mae_val}\n')
    print(f'test: MAE：{mae_test}\n')
    print(f'train: MAPE：{mape_train}\n')
    print(f'val: MAPE：{mape_val}\n')
    print(f'test: MAPE：{mape_test}\n')




