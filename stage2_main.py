# %%
import os
import math
import joblib
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import hmean
from model import *
from collections import namedtuple
from sklearn.preprocessing import MinMaxScaler, RobustScaler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 기존 계량과제의 Testset
test_a = pd.read_csv('../data/test/A_test_ver1.csv')
test_a['shift_nm'] = test_a['shift_nm'].replace({'N':1, 'D':2, 'A':3})
test_a = test_a.drop(['weekday.1'], axis=1)
test_a['weekday'] = test_a['weekday'].apply(int)
test_a = pd.get_dummies(test_a, columns=['weekday', 'shift_nm'])
total_test = test_a.drop(['time_t', 'sample'], axis=1)

# 추후 Reward정의할때 변수들간의 Scale을 고려하기 위해 (MinMax scaling -> 조화평균)
reward_scaler_dict = {}
target_y = ['sc_at03a_m_xq03', 'sc_at03b_xq01', 'sc_at07a_m_xq03', 'sc_at07b_xq01', 'fg_at32_xq01']
for i, column in enumerate(total_test.columns):
    if column in target_y:
        scaler = MinMaxScaler()
        scaler.fit(total_test[column].values.reshape(-1, 1))
        reward_scaler_dict[column] = scaler

# Train set에 적용한 Standard Scaler Test set에 적용
scaler_dict = joblib.load('std_scaler_dict')
for column in total_test.columns:
    scaler = scaler_dict[column]
    total_test.loc[:, column] = scaler.transform(total_test.loc[:, column].values.reshape(-1, 1))

# 변화시킬 2가지 x변수 및 컬럼에서 위치하는 순서 (313, 314번째) -> 이후 변화시킬때 사용하기위해
target_x = ['ai_ft01_xq01', 'ai_ft02_xq01']  
for i, column in enumerate(total_test.columns):
    if column in target_x:
        print(i)



# %%
# Agent 정의
Transition = namedtuple('Transition', ('state','action','next_state','reward'))
hidden_size = 128
num_layers = 1
class DQN_gru(nn.Module):
    def __init__(self,state_size, action_size):
        super(DQN_gru,self).__init__()
        self.main = nn.GRU(input_size=state_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, action_size)
    def forward(self, input):
        batch = input.shape[0]
        output, hidden = self.main(input)
        output = output[:,-1,:].reshape(batch,-1)
        output = self.linear(output)
        return output

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory)<self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position+1) % self.capacity
    
    def sample(self,batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class Agent : 
    def __init__(self, state_size, batch_size, eval_epoch=None):
        self.state_size = state_size
        self.action_size = 4
        self.memory = ReplayMemory(10000)
        self.inventory = []
        self.eval_epoch = eval_epoch
        
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95
        self.batch_size = batch_size
        
        # 평가때는 학습된 모델
        if self.eval_epoch is not None:
            self.policy_net = torch.load(f'../results/RL_model/policy_{self.eval_epoch}epoch.pth', map_location=device)
            self.target_net = torch.load(f'../results/RL_model/target_{self.eval_epoch}epoch.pth', map_location=device)
        else:
            self.policy_net = DQN_gru(self.state_size, self.action_size).to(device)
            self.target_net = DQN_gru(self.state_size, self.action_size).to(device)
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=0.005, momentum= 0.9)

    def act(self, state): 
        if not self.eval_epoch and np.random.rand() <= self.epsilon:
            # if self.epsilon > 0.1:
            #     self.epsilon -= 0.005
            return random.randrange(self.action_size)
        tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        options = self.target_net(tensor)
        return np.argmax(options[0].detach().cpu().numpy())

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return 
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        next_state = torch.as_tensor(batch.next_state[0], dtype=torch.float32, device=device)
        non_final_mask = torch.tensor(tuple(map(lambda s : s is not None, next_state))).to(device)
        non_final_next_states = next_state.clone()

        state_batch = torch.as_tensor(batch.state[0], dtype=torch.float32, device=device)
        action_batch = torch.LongTensor(batch.action).to(device)
        reward_batch = torch.FloatTensor(batch.reward).to(device)
        state_action_values = self.policy_net(state_batch).reshape((self.batch_size, self.action_size)).gather(1, action_batch.reshape((self.batch_size, 1)))
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1,1)
        self.optimizer.step()


# %%
# Stage1에서 사용한
encoder = Encoder(input_dim=total_test.shape[1], hidden_dim=256, n_layers=2, dropout=0, bid=False).to(device)
decoder = Decoder(output_dim=5, hidden_dim=256, n_layers=2, dropout=0, bid=False).to(device)
model = Seq2Seq(encoder, decoder, device=device, target_seq_len=10, teacher_force=0).to(device)
# 가중치 Load
model.load_state_dict(torch.load('../results/best_model.pth'))

# %%
def predict(tensor, model, scaler_dict, reward_scaler_dict):
    pred = model(tensor)
    pred = pred.detach().cpu().numpy().reshape(-1, 5)
    df_pred = pd.DataFrame(data=pred, columns = ['sc_at03a_m_xq03', 'sc_at03b_xq01', 'sc_at07a_m_xq03', 'sc_at07b_xq01', 'fg_at32_xq01'])
    minmax_df_pred = df_pred.copy()

    # Scaler사용한 값 원래대로 복원
    for column in df_pred.columns:
        scaler = scaler_dict[column]
        df_pred[column] = scaler.inverse_transform(df_pred[column].values.reshape(-1, 1))

    # Reward 조화평균 계산용 Minmax scaler
    for column in df_pred.columns:
        scaler = reward_scaler_dict[column]
        minmax_df_pred[column] = scaler.transform(df_pred[column].values.reshape(-1, 1))

    return df_pred, minmax_df_pred

# %%
# Agent 및 파라미터 정의
state_size = total_test.shape[1]
batch_size=1
agent = Agent(state_size=state_size, batch_size=batch_size)

# 통계량 (값을 얼마나 바꾸어야할지에 대한 insight)
total_test[target_x].iloc[:, 0].describe()
# 01은 분산 0.088
total_test[target_x].iloc[:, 1].describe()
# 02는 분산 0.72
X01_variation = 0.88 / 10
X02_variation = 0.72 / 10
# 분산의 1/6정도로 움직이도록

# Test하려는 Case
case1 = 696
case2 = 554
# case1과 case2에 해당하는 샘플 뽑아오기
case1_X_idx = np.array(test_a[test_a['sample'] == case1].index)
case2_X_idx = np.array(test_a[test_a['sample'] == case2].index)
case1_X = total_test.iloc[case1_X_idx].values
case2_X = total_test.iloc[case2_X_idx].values
case1_X.shape, case2_X.shape

case1_X = torch.as_tensor(case1_X, device=device, dtype=torch.float32).unsqueeze(0)   # shape:(1, 50, 343)
case2_X = torch.as_tensor(case2_X, device=device, dtype=torch.float32).unsqueeze(0)


# %%
def run(case_X, epoch):

    episode_count = epoch
    state = case_X.clone()
    pred, minmax_pred = predict(case_X, model, scaler_dict, reward_scaler_dict)  # 예측한 10sequence, 5columns(10, 5)
    init_pred = minmax_pred.copy()
    action_list = []
    visualize_X_list = []
    visualize_y_list = []
    total_profit = 0
    best_profit = 0
    x1_max_lim = 15
    x1_min_lim = -1.65
    x2_max_lim = 10
    x2_min_lim = -1.5
    '''
    투입 X, 최대, 최소값 (Standard Scaling된 것) -> 이 기준보다는 낮아지지않도록
    ai_ft01_xq01   6.958251, -1.621997
    ai_ft02_xq01   2.348127, -1.486343
    '''

    for e in range(1, episode_count + 1):
    #    print("Episode " + str(e) + "/" + str(episode_count))
        target_x = ['ai_ft01_xq01', 'ai_ft02_xq01']  

        action = agent.act(state)  # action
        next_pred, next_minmax_pred = predict(state, model, scaler_dict, reward_scaler_dict)  # 학습된 Seq2seq모델로 다음시점 예측
        
        # t와 t+1시점의 5개 y의 평균값 차이가 클수록 다음y가 줄어든것이므로 보상을 줌.
        # 변수별로 평균낸후 변수별 Scale차이를 보정하기위해 조화평균계산
        diff_mean = (minmax_pred - next_minmax_pred).mean(axis=0) + 1
        reward = hmean(diff_mean) - 1 # 모든 컬럼의 Y 조화평균값 (최적의상태저장)
        total_profit += reward

        # 초기값과 가장 차이가 큰 시점의 값을 저장하기위해
        save_profit = hmean((init_pred - next_minmax_pred).mean(axis=0) + 1) - 1
        # print(save_profit)

        next_state = state.clone()

        if action == 0:   # ai_ft01,02 모두 Up
            if (next_state[:,:,313].mean() > x1_max_lim) or (next_state[:,:,314].mean() > x2_max_lim):
                break
            next_state[:,:,313] += X01_variation   # ai_ft01_xq01
            next_state[:,:,314] += X02_variation   # ai_ft02_xq01
            action_list.append(action)
            visualize_X_list.append(next_state[0,:,313:315].detach().cpu().numpy().mean(axis=0))
            visualize_y_list.append(next_pred.mean().values)

        elif action == 1: # 01=Up, 02=Down
            if (next_state[:,:,313].mean() > x1_max_lim) or (next_state[:,:,314].mean() < x2_min_lim):
                break
            next_state[:,:,313] += X01_variation   # ai_ft01_xq01
            next_state[:,:,314] -= X02_variation / 1.5   # ai_ft02_xq01
            action_list.append(action)
            visualize_X_list.append(next_state[0,:,313:315].detach().cpu().numpy().mean(axis=0))
            visualize_y_list.append(next_pred.mean().values)
        elif action == 2: # 01=Down, 02=Up
            if (next_state[:,:,313].mean() < x1_min_lim) or (next_state[:,:,314].mean() > x2_max_lim):
                break
            next_state[:,:,313] -= X01_variation / 1.5   # ai_ft01_xq01
            next_state[:,:,314] += X02_variation   # ai_ft02_xq01
            action_list.append(action)
            visualize_X_list.append(next_state[0,:,313:315].detach().cpu().numpy().mean(axis=0))
            visualize_y_list.append(next_pred.mean().values)
        elif action == 3: # 01=Down, 02=Down
            if (next_state[:,:,313].mean() < x1_min_lim) or (next_state[:,:,314].mean() < x2_min_lim):
                break
            next_state[:,:,313] -= X01_variation / 1.5   # ai_ft01_xq01
            next_state[:,:,314] -= X02_variation / 1.5   # ai_ft02_xq01
            action_list.append(action)
            visualize_X_list.append(next_state[0,:,313:315].detach().cpu().numpy().mean(axis=0))
            visualize_y_list.append(next_pred.mean().values)

        agent.memory.push(state, action, next_state, reward)
        state = next_state.clone()
        minmax_pred = next_minmax_pred.copy()

        # Y의 조화평균이 가장 좋을때의 값저장
        if best_profit < save_profit:
            best_iter = e
            best_profit = save_profit
            x_column_values = pd.DataFrame(next_state[0,:,313:315].detach().cpu().numpy(), columns=target_x)   # (50, 2)
            for column in target_x:
                scaler = scaler_dict[column]
                x_column_values[column] = scaler.inverse_transform(x_column_values[column].values.reshape(-1, 1))
            y_column_values = next_pred   # (10, 5)
            
        agent.optimize()
    return visualize_X_list, visualize_y_list, x_column_values, y_column_values, best_iter, action_list

# %%
visualize_X_list, visualize_y_list, x_column_values, y_column_values, best_iter, action_list = run(case_X = case1_X, epoch=1000)

# 바뀐 X와 Y값
x_column_values.to_csv('../results/best_inputs.csv', index=False)
y_column_values.to_csv('../results/best_outputs.csv', index=False)
# %%
def visualize(visualize_X_list, visualize_y_list):
    target_x = ['ai_ft01_xq01', 'ai_ft02_xq01']  
    target_y = ['sc_at03a_m_xq03', 'sc_at03b_xq01', 'sc_at07a_m_xq03', 'sc_at07b_xq01', 'fg_at32_xq01']

    visualize_X_list = np.concatenate(visualize_X_list).reshape(-1, 2)
    visualize_X_list = pd.DataFrame(visualize_X_list, columns=target_x)
    # Scaler사용한 값 원래대로 복원
    for column in visualize_X_list.columns:
        scaler = scaler_dict[column]
        visualize_X_list[column] = scaler.inverse_transform(visualize_X_list[column].values.reshape(-1, 1))
    visualize_y_list = np.concatenate(visualize_y_list).reshape(-1, 5)

    c_dict = {0: 'red', 1: 'blue', 2: 'green', 3: 'black'}
    total_range = np.arange(len(visualize_y_list))
    action_dict = {0:'Up & Up', 1:'Up & Down', 2:'Down & Up', 3:'Down & Down'}

    f, ax = plt.subplots(7, 1, figsize=(15, 15))
    for axes in range(5):
        ax[axes].plot(total_range, visualize_y_list[:, axes])
        ax[axes].axvline(best_iter)
        ax[axes].set_title(target_y[axes], fontsize=15)
        ax[axes].xaxis.set_visible(False)
        for g in np.unique(action_list):
            i = np.where(action_list == g)[0]
            ax[axes].scatter(total_range[i], visualize_y_list[i, axes],
                            c=c_dict[g], label=action_dict[g], s=30)
    ax[0].legend(loc='upper right')
    ax[5].plot(total_range, visualize_X_list.iloc[:, 0], label='ft01', color='red')
    ax[6].plot(total_range, visualize_X_list.iloc[:, 1], label='ft02', color='red')
    ax[5].set_title(target_x[0], fontsize=15)
    ax[6].set_title(target_x[1], fontsize=15)
    ax[5].legend()
    ax[6].legend()
    ax[5].xaxis.set_visible(False)
    plt.subplots_adjust(hspace=0.35)
    # plt.savefig('../results/figure.png')
    plt.show()
# %%
visualize(visualize_X_list, visualize_y_list)
# %%
x_column_values.to_csv('../results/best_inputs.csv', index=False)
y_column_values.to_csv('../results/best_outputs.csv', index=False)