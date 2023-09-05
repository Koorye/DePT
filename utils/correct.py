import os
import os.path as osp
import pandas as pd


for root, dirs, filenames in os.walk('results'):
    for filename in filenames:
        if filename.endswith('.csv'):
            path = osp.join(root, filename)
            df = pd.read_csv(path)
        
            if 'H' not in df.columns or 'base_acc' not in df.columns or 'new_acc' not in df.columns:
                continue

            print(f'correct >>> {path}')
            df = df.drop(columns='H')
            df['H'] = 2 / (1 / df['base_acc'] + 1 / df['new_acc'])

            df.round(2).to_csv(path, index=None)
        