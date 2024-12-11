import shap
import torch
from model import get_model
from data import get_data
from matplotlib import pyplot as plt

x_train, x_val, x_test, y_train, y_val, y_test = get_data("pfas_nalv.xlsx", random=9204)
shap.initjs()


net = get_model()
net.load_state_dict(torch.load('checkpoint/save_model.ckpt'), False)
print(x_train.shape)

explainer = shap.DeepExplainer(net, x_train)
shap_values = explainer.shap_values(x_train)


shap.summary_plot(shap_values, x_train, max_display=8)
plt.show()
