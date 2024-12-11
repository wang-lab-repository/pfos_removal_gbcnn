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

# 模型预测
with torch.no_grad():
    predictions = mymodel(x_test).squeeze().numpy()
print(predictions)