from captum.attr import ShapleyValueSampling
from data3 import get_data, get_dataloader
import numpy as np
import matplotlib.pyplot as plt
import joblib
import torch
import math
from skorch import NeuralNetRegressor, NeuralNet
from model import get_model, get_1dcnn_model
import shap

# from featurecaptum.attr import IntegratedGradients
x_train, x_val, x_test, y_train, y_val, y_test = get_data("pfas_nalv - 副本.xlsx", random=9204)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)
y_val = np.ravel(y_val)

mymodel = get_1dcnn_model(7)
mymodel.load_state_dict(torch.load('checkpoint/1DGBCNN_model.pth'), False)
mymodel.eval()


plt.rcParams['font.family'] = 'serif'

plt.rcParams['font.serif'] = 'Times new Roman'

plt.rcParams['font.size'] = 13


# plt.subplots_adjust(left=0.35, right=1.0, top=0.88, bottom=0.15)


features = ['MWCO(Da)', 'Pore size(nm)', 'water flux(LMH)', 'Temperature (˚C)', 'PFOS con (ppb)', 'pH',
                    'Pressure (MPa)', 'Divalent cations (mmol/L)',
                    'Monovalent cations (mmol/L)', 'Trivalent cations (mmol/L)',
            'type__ESNA1-K1', 'type__HYDRACORE', 'type__NE70',
            'type__NF270', 'type__PMIA', 'type__Poly NF']
captumshap = ShapleyValueSampling(mymodel)
print(captumshap)
attributions = captumshap.attribute(x_train, n_samples=300)  # 200
print(x_train)
print(attributions)
attributions_numpy = attributions.squeeze().cpu().detach().numpy()
# print(attributions_numpy)
# shap.summary_plot(attributions_numpy, x_train, feature_names=features, plot_size=(9, 6.5))
#
#
# # plt.savefig('featureShap.tif', format='tif')
#
#
x_train_np = x_train.numpy()
x_test_np = x_test.numpy()
# # shap.dependence_plot('MWCO(Da)', attributions_numpy, x_train_np, feature_names=features,
# #                      interaction_index='PFOS con (ppb)', show=False)
# # plt.xlabel('MWCO')
# # plt.show()
#
shap.dependence_plot('MWCO(Da)', attributions_numpy, x_train_np, feature_names=features, interaction_index='Pore size(nm)')

shap.dependence_plot('Temperature (˚C)', attributions_numpy, x_train_np, feature_names=features,
                     interaction_index=None)
shap.dependence_plot('Pressure (MPa)', attributions_numpy, x_train_np, feature_names=features,
                     interaction_index=None)
shap.dependence_plot('Pressure (MPa)', attributions_numpy, x_train_np, feature_names=features,
                     interaction_index='water flux(LMH)')
#
# # 对于阳离子，可以根据实际情况选择最相关的进行比较
# shap.dependence_plot('Divalent cations (mmol/L)', attributions_numpy, x_train_np, feature_names=features,
#                      interaction_index=None)
#
# # 对于阳离子，可以根据实际情况选择最相关的进行比较
# shap.dependence_plot('Divalent cations (mmol/L)', attributions_numpy, x_train_np, feature_names=features,
#                      interaction_index='Monovalent cations (mmol/L)')


import shap

# # 确保输入数据是 PyTorch 张量并且形状正确
# # x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)  # 添加通道维度
# # x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)   # 添加通道维度
# # 构建 shap解释器
# x_train_np = x_train.detach().cpu().numpy()
# print(x_train.shape)
# explainer = shap.GradientExplainer(mymodel, x_train)
#
# # 计算测试集的shap值
#
# # shap_values = explainer.shap_values(x_train)
#
# # shap.summary_plot(shap_values, x_train, feature_names=features, plot_type="dot")
#
# shap_interaction_values = explainer.shap_interaction_values(x_train.numpy())
#
# shap.summary_plot(shap_interaction_values, x_train.numpy())


# 创建 shap.Explanation 对象

# explainer = shap.DeepExplainer(mymodel, x_train)
# # shap_values = explainer.shap_values(x_train)
#
# # shap.plots.heatmap(shap_values)
#
# shap_explanation = shap.Explanation(values=attributions_numpy,
#
#                                     base_values=explainer.expected_value,
#
#                                     data=x_train, feature_names=features)
#
# # 绘制热图
# # 计算每个实例的总 SHAP 值并获取排序索引
# order = np.argsort(attributions_numpy.sum(1))
# plt.rcParams['figure.figsize'] = (12, 10)
# plt.tight_layout()
# shap.plots.heatmap(shap_explanation, instance_order=order)
