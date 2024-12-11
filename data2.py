import pandas as pd
from sklearn.model_selection import train_test_split
from sampling import get_balance_dataset
from utils import make_index, generate_o_d
import torch
import torch.utils.data as Data


def get_data(path, random, percent=1):
    columns_list = ['Membrane type', 'Temperature (˚C)', 'PFOS con (ppb)', 'pH',
                    'Pressure (MPa)', 'Divalent cations (mmol/L)',
                    'Monovalent cations (mmol/L)', 'Trivalent cations (mmol/L)',
                    'PFOS rejection (%)']
    df_all = pd.read_excel(path)
    # print("df_all的类型：", type(df_all))
    # print(len(df_all))
    # print(df_all.info())
    # 从数据集中随机抽取25%的行
    # df_all = df_all.iloc[:int(len(df_all) * percent)]
    # df_all = df_all.head(int(len(df_all) * percent))
    # df_all = df_all.sample(frac=percent, random_state=1)
    # print("df_all的类型：", type(df_all))
    # print(len(df_all))
    # print(df_all.info())
    MemType = {1: 'NF270', 2: 'HYDRACORE', 3: 'PMIA', 4: 'NE70', 5: 'Poly(piperazineamide) NF', 6: 'ESNA1-K1'}
    Yc = ['PFOS rejection (%)']
    y = df_all[Yc]
    x = df_all.drop(['Data', 'PFOS rejection (%)'], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.10, random_state=random)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, shuffle=True, test_size=0.111,
                                                      random_state=random)
    temp_df = pd.concat([x_train, y_train], axis=1)
    # temp_df=make_index(temp_df)
    train_data = pd.DataFrame(temp_df, columns=columns_list)
    data_sampled = get_balance_dataset(data=train_data, column='Membrane type', random=random,
                                      columns_list=columns_list)
    #data_sampled = train_data
    # print("data_sampled:-----------", data_sampled.info())
    x_train = data_sampled.iloc[:, 0:8]
    y_train = data_sampled.iloc[:, 8:9]
    # print("-----------##1###-------", x_train)
    x_test = x_test[x_train.columns]
    x_val = x_val[x_train.columns]
    len_train = x_train.shape[0]
    len_val = x_val.shape[0]
    len_test = x_test.shape[0]
    x_new = pd.concat([x_train, x_val, x_test], axis=0)
    x_new = make_index(x_new)

    x_new = generate_o_d(MemType, 'Membrane type', x_new)
    # print("-----------##x_new1###-------", x_new)
    x_new = pd.get_dummies(x_new)
    # print("-----------##x_new2###-------", x_new)
    std = x_new.std()
    mean = x_new.mean()
    x_new = (x_new - mean) / std
    x_train = x_new.iloc[0:len_train, :]
    # print("-----------##2###-------", x_train)
    x_val = x_new.iloc[len_train:len_train + len_val, :]
    x_test = x_new.iloc[len_train + len_val:len_train + len_val + len_test, :]


    x_train = x_train.head(int(len(x_train) * percent))
    y_train = y_train.head(int(len(y_train) * percent))
    x_test = x_test.head(int(len(x_test) * percent))
    y_test = y_test.head(int(len(y_test) * percent))
    x_val = x_val.head(int(len(x_val) * percent))
    y_val = y_val.head(int(len(y_val) * percent))
    print(f"训练：{x_train.shape},验证：{x_val.shape}，测试：{x_test.shape}")

    x_train = torch.from_numpy(x_train.values).float()
    y_train = torch.from_numpy(y_train.values).float()
    x_test = torch.from_numpy(x_test.values).float()
    y_test = torch.from_numpy(y_test.values).float()
    x_val = torch.from_numpy(x_val.values).float()
    y_val = torch.from_numpy(y_val.values).float()

    return x_train, x_val, x_test, y_train, y_val, y_test


def get_dataloader(x_train, y_train):
    train_data = Data.TensorDataset(x_train, y_train)
    train_load = Data.DataLoader(
        dataset=train_data,
        batch_size=1024,
        shuffle=True,
    )
    return train_load
