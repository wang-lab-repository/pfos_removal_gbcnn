from model import get_model
from utils import mix_seed
import torch
from early_stopping import EarlyStopping
from data3 import get_data, get_dataloader
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


def train(is_evaluate=True):
    x_train, x_val, x_test, y_train, y_val, y_test = get_data("pfas_nalv - 副本.xlsx", random=9204, percent=1)
    train_load = get_dataloader(x_train, y_train)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"训练设备：{device}")
    mix_seed(9204)
    # mix_seed(77)
    epoch = 5000
    lr = 0.05
    lr_min = 5e-05
    step_size = 35
    net = get_model()
    if is_evaluate:
        net.load_state_dict(torch.load('checkpoint/save_model.ckpt'), False)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    optimizer1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size, eta_min=lr_min)
    loss_func = torch.nn.MSELoss()
    early_stopping = EarlyStopping(patience=300, delta=0)
    print(early_stopping.counter)

    train_loss_list = []
    test_loss_list = []
    if not is_evaluate:
        for i in range(epoch):
            net.train()
            for step, (train_x, train_y) in enumerate(train_load):
                train_pre = net(train_x)
                train_loss = loss_func(train_pre, train_y)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                print("第%d个epoch的学习率：%f" % (i, optimizer.param_groups[0]['lr']))
                # optimizer1.step()


            prediction_val = net(x_val)
            R2_val = 1 - torch.mean((y_val - prediction_val) ** 2) / torch.mean(
                (y_val - torch.mean(y_val)) ** 2)
            early_stopping(-R2_val)

            net.eval()
            train_loss_list.append(np.sqrt(mean_squared_error(y_train.detach().numpy(), net(x_train).detach().numpy())))
            test_loss_list.append(np.sqrt(mean_squared_error(y_test.detach().numpy(), net(x_test).detach().numpy())))

            if early_stopping.early_stop:
                print("Early stopping")
                break
    net.eval()

    prediction_train = net(x_train)
    prediction_val = net(x_val)
    prediction_test = net(x_test)

    plt.plot(train_loss_list, label='Training loss')
    plt.plot(test_loss_list, label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Test Loss')
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
