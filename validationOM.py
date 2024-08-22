import random
import seaborn as sns
import torch
from tools.ReadData import readOpenBmi
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from model.MIFNet import MIFNet
root='/root/autodl-tmp/pythonProject2/Data/openbmi/'


model=MIFNet().cuda()

subacc=np.zeros(54)
subkappa=np.zeros(54)
cumulative_cm=np.zeros((2,2))
for i in range(54):
    SEED = 2024
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    nSub=i+1
# 加载模型权重
    model.load_state_dict(torch.load(f'/root/autodl-tmp/pythonProject2/BP/256_s_om/{nSub}model.pth'))
    model.eval()
    _, _,test_data,test_label = readOpenBmi(root,nSub,True,[(4,16),(16,40)])
    test_data=torch.from_numpy(test_data).cuda()
    test_label=torch.from_numpy(test_label).cuda()-1
    # validation
    Cls = model(test_data.float())
    y_pred = torch.max(Cls, 1)[1]
    acc = float((y_pred == test_label).cpu().numpy().astype(int).sum()) / float(test_label.size(0))

    # confusion matrixx
    cm = confusion_matrix(test_label.cpu().numpy(), y_pred.cpu().numpy())
    cumulative_cm += cm
    print(f'{nSub}accuracy is {acc}')
    subacc[i] = acc

    kappa = cohen_kappa_score(test_label.cpu().numpy(), y_pred.cpu().numpy())
    print(f'{nSub} kappa is {kappa}')
    subkappa[i] = kappa

print(np.mean(subacc))
print(np.mean(subkappa))
print("Cumulative Confusion Matrix:")
print(cumulative_cm)


