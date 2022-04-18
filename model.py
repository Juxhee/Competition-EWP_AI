# %%
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout, bid:bool):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=bid)
    
    def forward(self, input_seq):
        outputs, (hidden, cell) = self.rnn(input_seq)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, n_layers, dropout, bid:bool):
        super().__init__()
        '''
        output_dim: 예측할 Y변수 개수(5)
        '''
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(output_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=bid)
        if bid:
            self.fc = nn.Linear(hidden_dim*2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, input, hidden, cell):
        '''
        input: (B, 1, input_dim) = (B, 1, 5)
        hidden, cell: (n_layers*Bid, B, hidden_dim) = (1, B, 512)
        '''
        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        # output: (B, 1, hidden_dim)
        # hidden,cell: (n_layers*bid, B, hidden_dim)

        pred = self.fc(output)   
        # pred:(B, 1, output_dim) = (B, 1, 5)

        return pred, hidden, cell
        # pred는 최종 예측할 5개의 Y값
        # output은 input X의 dimension을 그대로 갖는 Vector


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, target_seq_len:int=10, teacher_force=0.5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_seq_len = target_seq_len
        self.teacher_force = teacher_force
        self.device= device

    def forward(self, input_seq, target_seq=None):
        '''
        input_seq: (B, input_seq_len, input_dim)
        target_seq: (B, target_seq_len, target_ydim)
        '''
        batch_size = input_seq.shape[0]
        target_len = self.target_seq_len
        target_ydim = self.decoder.output_dim   # 5
        
        # 단계마다의 예측을 담아둘 Vector
        outputs = torch.zeros(batch_size, target_len, target_ydim).to(self.device)
        hidden, cell = self.encoder(input_seq)

        # Decoder에 넣을 초기값 생성 (이부분 좀 더 Search 필요)
        input = torch.ones((batch_size, 1, target_ydim)).to(self.device)
        # input = torch.Tensor(np.repeat(-1, batch_size))

        # Encoder의 hidden,cell을 받아 Decoder에서 한 Time step씩 출력
        for t in range(0, target_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            # output: (B, 1, output_dim) = (B, 1, 5)

            outputs[:, t] = output.squeeze()
            
            # Teacher forcing 여부
            # Teacher forcing이 True이면 Target을 다음 Input으로 넘겨주고
            # False이면 이전 Step의 예측으로 나온 벡터를 Input으로 넘겨줌.
            if self.teacher_force:
                teacher_forcing = random.random() < self.teacher_force
                input = target_seq[:, t].unsqueeze(1) if teacher_forcing else output
            else:
                input = output.clone()

        return outputs   # (B, target_len, target_ydim)


class ManytoMany(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, dropout, device, target_seq_len:int=10, teacher_force=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.target_seq_len = target_seq_len
        self.teacher_force = teacher_force
        self.device = device
        self.rnn = nn.LSTM(input_dim, input_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, input_seq, target_seq=None):
        batch_size = input_seq.shape[0]
        target_len = self.target_seq_len
        target_ydim = self.output_dim

        # 단계마다의 예측을 담아둘 Vector
        outputs = torch.zeros(batch_size, target_len, target_ydim).to(self.device)

        # 기존 50시퀀스 먼저학습
        output, (hidden, cell) = self.rnn(input_seq)
        # 예측부분 첫 INPUT = 학습부분 마지막 Output = t50의 output 벡터
        input = output[:, -1, :].unsqueeze(1)  # (B, 1, input_dim)

        for t in range(0, target_len):
            input_seq = torch.cat([input_seq, input], dim=1)  # (B, 50+1, input_dim)
            output, (hidden, cell) = self.rnn(input_seq, (hidden, cell))   # output shape: (B, 50+t, input_dim)
            preds = self.fc(output[:, -1, :])   # 마지막 예측값을 fc에 태우기
            outputs[:, t] = preds  # preds: (B, 5)
            # outputs: (B, target_len, output_dim) = (B, 10, 5)

            input = output.clone()
            
            # Teacher forcing 여부
            # Teacher forcing이 True이면 Target을 다음 Input으로 넘겨주고
            # False이면 이전 Step의 예측으로 나온 벡터를 Input으로 넘겨줌.
            # if self.teacher_force:
            #     teacher_forcing = random.random() < self.teacher_force
            #     input = target_seq[:, t].unsqueeze(1) if teacher_forcing else output
            # else:
            

        return outputs

# %%
