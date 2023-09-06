# a tool for sorting out the results of experiments, you can ignore it
import os
import os.path as osp
import pandas as pd


base_dir = 'results/shots/'
dfs = []

for dir_ in os.listdir(base_dir):
    if not osp.isdir(osp.join(base_dir, dir_)):
        continue

    filename = os.listdir(osp.join(base_dir, dir_))[0]
    df = pd.read_csv(osp.join(base_dir, dir_, filename))
    df['method'], df['shot'] = dir_.split('_')
    df['shot'] = df['shot'].apply(lambda x: int(x[0]))
    dfs.append(df)
    
df = pd.concat(dfs).reset_index(drop=True)
df = df[df['dataset'] == 'average'].drop(columns='dataset').reset_index(drop=True)
df = df[['shot', 'method', 'base_acc', 'new_acc', 'H']]

df['method'] = df['method'].replace({
    'coop': 'CoOp',
    'cocoop': 'CoCoOp',
    'kgcoop': 'KgCoOp',
    'maple': 'MaPLe',
    'elpcoop': 'CoOp w/ DePT',
    'elpcocoop': 'CoCoOp w/ DePT',
    'elpkgcoop': 'KgCoOp w/ DePT',
    'elpmaple': 'MaPLe w/ DePT',
})

df = df.sort_values(['method', 'shot']).reset_index(drop=True)
df.to_csv(osp.join(base_dir, 'shots.csv'), index=None)


base_dir = 'results/epochs/'
dfs = []

for dir_ in os.listdir(base_dir):
    if not osp.isdir(osp.join(base_dir, dir_)):
        continue

    filenames = os.listdir(osp.join(base_dir, dir_))
    for filename in filenames:
        df = pd.read_csv(osp.join(base_dir, dir_, filename))
        df['method'], df['epoch'] = filename.split('.')[0].split('-')
        df['epoch'] = df['epoch'].apply(lambda x: int(x[2:]))
        dfs.append(df)
    
df = pd.concat(dfs).reset_index(drop=True)
df = df[df['dataset'] == 'average'].drop(columns='dataset').reset_index(drop=True)
df = df[['epoch', 'method', 'base_acc', 'new_acc', 'H']]

df['method'] = df['method'].replace({
    'coop': 'CoOp',
    'elpcoop': 'CoOp w/ DePT',
})

df = df.sort_values('epoch').reset_index(drop=True)
df.to_csv(osp.join(base_dir, 'epochs.csv'), index=None)
