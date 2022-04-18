# %%
import os
import gc
import time
import joblib
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader

from utils import *
from dataset import Task1_dataset, Task1_test_dataset
from model import Encoder, Decoder, Seq2Seq, ManytoMany

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# make_testset.py로 만들어진 파일 Load
test_a = pd.read_csv('../data/test/A_test_ver1.csv')
test_b = pd.read_csv('../data/test/B_test_ver1.csv')
sub = pd.read_csv('../data/test/task1_output_sample.csv')

test_a['shift_nm'] = test_a['shift_nm'].replace({'N':1, 'D':2, 'A':3})
test_b['shift_nm'] = test_b['shift_nm'].replace({'N':1, 'D':2, 'A':3})

test_a = test_a.drop(['weekday.1'], axis=1)
test_b = test_b.drop(['weekday.1'], axis=1)

test_a['weekday'] = test_a['weekday'].apply(int)
test_b['weekday'] = test_b['weekday'].apply(int)
test_a = pd.get_dummies(test_a, columns=['weekday', 'shift_nm'])
test_b = pd.get_dummies(test_b, columns=['weekday', 'shift_nm'])

total_test = pd.concat([test_a, test_b], axis=0, ignore_index=True)
total_test = total_test.drop(['time_t', 'sample'], axis=1)

# 일단 Null values가 60250이후 index만 존재 = B셋이므로 3.000898로 Fill
total_test['sc_at06_xq01'] = np.where(total_test['sc_at06_xq01'].isnull(), 3.000898, total_test['sc_at06_xq01'])

# %%
# Train set에 적용한 Standard Scaler Test set에 적용
scaler_dict = joblib.load('std_scaler_dict')
for column in total_test.columns:
    scaler = scaler_dict[column]
    total_test.loc[:, column] = scaler.transform(total_test.loc[:, column].values.reshape(-1, 1))

# %%
def predict(test_loader, model):
    model.eval()
    model.teacher_force=0
    pred_list = []
    for i, input_seq in enumerate(test_loader):
        input_seq = torch.tensor(input_seq, device=device, dtype=torch.float32)

        pred_seq = model(input_seq, target_seq=None)
        pred_list.append(pred_seq.detach().cpu().numpy())
        del input_seq, pred_seq
        gc.collect()
    pred_list = np.concatenate(pred_list).reshape(-1, 5)
    return pred_list
#%%
# 학습 때 사용했던 Hyper-Parameter
batch_size=32
window_size=50
num_workers=0
hidden_dim=256
n_layers=2
bid=False
dropout=0
teacher_force=0
test_dataset = Task1_test_dataset(total_test, window_size=window_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
print('Testset 길이:',len(test_dataset))

encoder = Encoder(input_dim=total_test.shape[1], hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout, bid=bid).to(device)
decoder = Decoder(output_dim=5, hidden_dim=hidden_dim, n_layers=n_layers, dropout=dropout, bid=bid).to(device)
model = Seq2Seq(encoder, decoder, device=device, target_seq_len=10, teacher_force=teacher_force).to(device)

# 학습된 가중치 Load
model.load_state_dict(torch.load('../results/best_model.pth'))
pred_list = predict(test_loader, model)

df_pred = pd.DataFrame(data=pred_list, columns = ['sc_at03a_m_xq03', 'sc_at03b_xq01', 'sc_at07a_m_xq03', 'sc_at07b_xq01', 'fg_at32_xq01'])

# Scaler사용한 값 원래대로 복원
for column in df_pred.columns:
    scaler = scaler_dict[column]
    df_pred[column] = scaler.inverse_transform(df_pred[column].values.reshape(-1, 1))

sub.iloc[:, 3:] = np.array(df_pred.values, dtype=np.float64)
sub['weekday'] = sub['weekday'].apply(int)
sub.to_csv('../results/4/task1_final.csv', index=False)
