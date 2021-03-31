import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

input_file = 'dqn_random'

file1 = open(input_file + '.txt', 'r')
lines = file1.read().split('\n')
to_process = lines[4:-7]
sample = to_process[1]
parts = sample.split(',')

record = dict()

for sample in to_process:
    parts = sample.split(',')
    for i, p in enumerate(parts):
        #print('p', p)
        if len(p.strip().split(':')) < 2 or p == 'done':
            continue
        if i == 0:
            # get rid of initial numbers
            p = p.split(':', 1)[1]


        label, val = tuple(p.split(':'))

        label = label.strip()
        val = val.strip().split(' ')[0].replace('s', '')

        if val == '--':
            val = -1.0
        else:
            val = float(val)

        record[label] = record.get(label, []) + [val]

# print('record[mean reward]', record['mean reward'][:5])

df = pd.DataFrame.from_dict(record)
print(df.head())
names = ['episode reward', 'mean reward', 'mean_q', 'mae']
for i, col_name in enumerate(names):
    plt.figure(i)
    plt.plot(df[col_name])
    plt.xlabel('episode')
    plt.ylabel(col_name)
    plt.savefig(input_file + '_' + col_name + '.jpg')

#print('lines 1-5', lines[4:-7])

# to -7


