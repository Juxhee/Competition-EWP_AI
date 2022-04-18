import torch
from torch.utils.data import Dataset, DataLoader

class Task1_dataset(Dataset):
    def __init__(self, df, window_size, predict_size):
        y_columns = ['sc_at03a_m_xq03', 'sc_at03b_xq01', 'sc_at07a_m_xq03', 'sc_at07b_xq01', 'fg_at32_xq01']
        self.X = df.values
        self.Y = df[y_columns].values
        self.window_size = window_size
        self.predict_size = predict_size

    def __getitem__(self, idx):
        input_seq = self.X[idx : idx+self.window_size]
        target_seq = self.Y[idx + self.window_size : idx + self.window_size + self.predict_size]
        return torch.from_numpy(input_seq), torch.from_numpy(target_seq)

    def __len__(self):
        return len(self.X) - self.window_size - self.predict_size + 1


class Task1_test_dataset(Dataset):
    def __init__(self, df, window_size):
        self.X = df.values
        self.window_size = window_size

    def __getitem__(self, idx):
        input_seq = self.X[self.window_size*idx : self.window_size*(idx+1)]
        return torch.from_numpy(input_seq)

    def __len__(self):
        return int(len(self.X) / self.window_size)