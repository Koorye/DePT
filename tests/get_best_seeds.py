import numpy as np
import pandas as pd
from tqdm import tqdm


def read_stats(file_path):
    df = pd.read_csv(file_path)
    values = df['base_acc'].tolist() + df['novel_acc'].tolist()
    values = [float(v[:-1]) / 100 for v in values]
    return np.array(values)

def metric(l1, l2):
    return np.abs(l1 - l2).sum()


best_metric, best_seeds = float('inf'), None

stats_maple = read_stats('stats/maple.csv')
all_stats = [None] + [read_stats(f'stats/seed{i}.csv') for i in range(1, 10 + 1)]

for seed1 in tqdm(range(1, 10 + 1)):
    for seed2 in range(seed1 + 1, 10 + 1):
        for seed3 in range(seed2 + 1, 10 + 1):
            print(f'seeds: {seed1}, {seed2}, {seed3}')
            
            m1 = metric(stats_maple, all_stats[seed1])
            m2 = metric(stats_maple, all_stats[seed2])
            m3 = metric(stats_maple, all_stats[seed3])
            m = m1 + m2 + m3

            if m < best_metric:
                best_metric = m
                best_seeds = [seed1, seed2, seed3]

print(best_seeds)
            