#%%
import os
import gc
import dateutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from tqdm import tqdm, tqdm_notebook

# %%
test_a = pd.read_csv('../data/test/A_test_input.csv')
test_b = pd.read_csv('../data/test/B_test_input.csv')
test_a_coal = pd.read_csv('../data/test/A_test_input_coal_data.csv')
test_b_coal = pd.read_csv('../data/test/B_test_input_coal_data.csv')

sub = pd.read_csv('../data/test/task1_output_sample.csv')

# %%
# a와 a_coal dataset 병합
total_a = pd.concat([test_a, test_a_coal.iloc[:, 2:]], axis=1)
total_b = pd.concat([test_b, test_b_coal.iloc[:, 2:]], axis=1)

# %%
# Description의 삭제컬럼
codebook = pd.read_csv('../T1_code_book.csv')
remove_columns = codebook[codebook.iloc[:, 5] == '삭제']['변수명'].apply(lambda x:x[2:]).values
total_a = total_a.drop(remove_columns, axis=1)
total_b = total_b.drop(remove_columns, axis=1)
print('삭제한 컬럼 개수:', len(remove_columns))  # 19개
# %%
# Missing value remove
null_columns = pd.read_csv('../data/null_columns.csv')['null']
print(len(null_columns))
total_a = total_a.drop(null_columns, axis=1)
total_b = total_b.drop(null_columns, axis=1)
total_a.to_csv('../data/test/A_test_ver1.csv', index=False)
total_b.to_csv('../data/test/B_test_ver1.csv', index=False)

# %%
