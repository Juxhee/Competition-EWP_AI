import os

def save_settings(args, save_path):
    with open(os.path.join(save_path, 'setting_info.txt'), 'w') as f:
        for key, value in args.__dict__.items():
            f.write(key + ' : ' + str(value) + '\n')


def write_logs(epoch, train_mae, val_mae, save_path):
    with open(os.path.join(save_path, 'setting_info.txt'), 'a') as f:
        f.write(f'{epoch}Epoch - Train MAE:{train_mae:.4f} | Valid MAE:{val_mae:.4f}\n')
