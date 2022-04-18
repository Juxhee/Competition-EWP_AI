#%%
import os
import gc
import time
import joblib
import warnings
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader

from utils import *
from dataset import Task1_dataset
from model import Encoder, Decoder, Seq2Seq, ManytoMany

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# %%
def train(train_loader, model, criterion, optimizer):
    model.train()
    model.teacher_force=0.5
    train_loss = 0
    train_mse = 0
    train_mae = 0
    best_loss = np.inf
    
    for i, (input_seq, target_seq) in enumerate(tqdm(train_loader)):
        input_seq = torch.tensor(input_seq, device=device, dtype=torch.float32)
        target_seq = torch.tensor(target_seq, device=device, dtype=torch.float32)

        optimizer.zero_grad()
        pred_seq = model(input_seq, target_seq)
        loss = criterion(pred_seq, target_seq)
        loss.backward()
        optimizer.step()

        # Log
        train_loss += loss.item()
        train_mse += torch.mean(torch.abs(pred_seq - target_seq)**2).item()
        del input_seq, target_seq, pred_seq
        gc.collect()
    train_loss /= len(train_loader)
    train_mse /= len(train_loader)
    print(f'Train Loss:{train_loss:.4f} | MSE:{train_mse:.4f}')
    return train_loss, train_mse

def validate(val_loader, model, criterion):
    model.eval()
    model.teacher_force=0
    val_loss = 0
    val_mse = 0
    val_mae = 0
    best_loss = np.inf
    
    with torch.no_grad():
        for i, (input_seq, target_seq) in enumerate(val_loader):
            input_seq = torch.tensor(input_seq, device=device, dtype=torch.float32)
            target_seq = torch.tensor(target_seq, device=device, dtype=torch.float32)

            pred_seq = model(input_seq, target_seq=None)
            loss = criterion(pred_seq, target_seq)

            # Log
            val_loss += loss.item()
            val_mse += torch.mean(torch.abs(pred_seq - target_seq)**2).item()
            del input_seq, target_seq, pred_seq
            gc.collect()
        val_loss /= len(val_loader)
        val_mse /= len(val_loader)
        print(f'Val Loss:{val_loss:.4f} | MSE:{val_mse:.4f}')
    return val_loss, val_mse


# %%
# data load
train_a = pd.read_csv('../data/train/A_train_ver1.csv')
train_b = pd.read_csv('../data/train/B_train_ver1.csv')

train_a['shift_nm'] = train_a['shift_nm'].replace({'N':1, 'D':2, 'A':3})
train_b['shift_nm'] = train_b['shift_nm'].replace({'N':1, 'D':2, 'A':3})

total_train = pd.concat([train_a, train_b])
total_train = pd.get_dummies(total_train, columns=['weekday', 'shift_nm'])
total_train = total_train.drop(['time_t'], axis=1)

# 변수값 Scaling
scaler_dict = {}
for column in total_train.columns:
    scaler = StandardScaler()
    total_train.loc[:, column] = scaler.fit_transform(total_train.loc[:, column].values.reshape(-1, 1))
    scaler_dict[column] = scaler

joblib.dump(scaler_dict, 'std_scaler_dict')

# %%
def main(args):
    start = time.time()
    save_path = os.path.join(args.model_path, args.experiment_num)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        if len(os.listdir(save_path))>1:
            print('Create New Folder')
            raise ValueError
        else:
            pass

    # Parameter 기록
    save_settings(args, save_path)

    dataset = Task1_dataset(total_train, window_size=args.window_size, predict_size=10)
    total_data = len(dataset)   # 462191
    train_dataset, val_dataset = random_split(dataset, [int(total_data*0.8), total_data-int(total_data*0.8)])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(len(train_dataset), len(val_dataset))

    if args.arch == 'seq2seq':
        encoder = Encoder(input_dim=total_train.shape[1], hidden_dim=args.hidden_dim, n_layers=args.n_layers, dropout=args.dropout, bid=args.bid).to(device)
        decoder = Decoder(output_dim=5, hidden_dim=args.hidden_dim, n_layers=args.n_layers, dropout=args.dropout, bid=args.bid).to(device)
        model = Seq2Seq(encoder, decoder, device=device, target_seq_len=10, teacher_force=args.teacher_force).to(device)
    elif args.arch == 'manytomany':
        model = ManytoMany(input_dim=total_train.shape[1], hidden_dim=args.hidden_dim, output_dim=5, n_layers=args.n_layers, dropout=args.dropout, device=device, target_seq_len=10, teacher_force=args.teacher_force)

    criterion = nn.L1Loss()

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
    elif args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.5, patience=5)

    # Training, Validate
    best_loss = np.inf
    for epoch in range(1, args.num_epochs + 1):
        print(f'{epoch}Epoch')
        train_loss, train_mse = train(train_loader, model, criterion, optimizer)
        val_loss, val_mse = validate(val_loader, model, criterion)
        scheduler.step(val_loss)
        # Save Models
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_path, 'best_model.pth'))

        torch.save(model.state_dict(), os.path.join(save_path, f'{epoch}epoch.pth'))
        write_logs(epoch, train_loss, val_loss, save_path)
    end = time.time()
    print(f'Total Process time:{(end-start)/60:.3f}Minute')
    print(f'Best Epoch:{best_epoch} | MAE:{best_loss:.4f}')

    

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_num', type=str, default='7')
    parser.add_argument('--info', type=str, default='std scaler, only A dataset')
    parser.add_argument('--arch', type=str, default='seq2seq')
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=180)
    parser.add_argument('--window_size', type=int, default=50)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--bid', type=bool, default=False)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--teacher_force', type=float, default=0.5)
    parser.add_argument('--lr', type=int, default=0.001)
    parser.add_argument('--clipping', type=int, default=0.7)
    parser.add_argument('--optimizer', type=str, default='AdamW')
#    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--model_path', type=str, default='../results/')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(args)
    main(args)
    
