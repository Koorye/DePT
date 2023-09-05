import math
import numpy as np
import os
import os.path as osp
import pandas as pd
import patchworklib as pw
import pickle
import shutil
import torch
from plotnine import *
from tqdm import tqdm


def read_feats(root, trainer, dataset, seed):
    filename = f'feats_{trainer}_{dataset}_seed{seed}.pkl'
    path = osp.join(root, filename)
    with open(path, 'rb') as f:
        feats = pickle.load(f)
    return feats

def filter_labels(feats, labels_reserved):
    labels = feats['label'].copy()
    text_feats = feats['text'].copy()
    img_feats = feats['img'].copy()

    flag = [label in labels_reserved for label in labels]
    labels = labels[flag]
    img_feats = img_feats[flag]
    text_feats = text_feats[labels_reserved]

    label_to_new_label = {label: idx for idx, label in enumerate(labels_reserved)}
    labels = np.array([label_to_new_label[label] for label in labels])

    feats = dict(
        label=labels,
        text=text_feats,
        img=img_feats)

    return feats

def cal_importance_per_channel(feats, channel_inds):
    labels = torch.from_numpy(feats['label']).long().cuda()
    img_feats = torch.from_numpy(feats['img']).float().cuda()
    text_feats = torch.from_numpy(feats['text']).float().cuda()
    
    img_feats = img_feats[:, channel_inds]
    text_feats = text_feats[:, channel_inds]
    # (N, D), (C, D) -> (N, D), (D, C) -> (N, C)
    similarities = img_feats @ text_feats.t()
    similarities = similarities.clamp_min_(0.)
    # (N, C) -> (N,)    
    similarities_gt = torch.gather(similarities, 1, labels.unsqueeze(-1)).squeeze(-1)
    # (N, C) -> (N,)
    # (N,), (N,) -> (N,) -> scaler
    importance = similarities_gt / (similarities.mean(dim=1) + 1e-12)
    return importance.mean().item()

def cal_importance(feats, interval):
    importances = []
    for channel_ind_start in tqdm(range(0, 512)):
        channel_ind_end = channel_ind_start + interval
        channel_inds = list(range(channel_ind_start, channel_ind_end))
        
        channel_inds = [ind - 512 if ind >= 512 else ind for ind in channel_inds]
        
        importance = cal_importance_per_channel(feats, channel_inds)
        importances.append(importance)
        
    return np.array(importances)



ROOT = 'stats/'
TRAINERS = ['OracleStats', 'CoOpStats', 'ExtrasLinearProbeCoOpStats']
DATASETS = ['Caltech101', 'DescribableTextures', 'EuroSAT', 'FGVCAircraft', 'Food101', 'OxfordFlowers', 'OxfordPets', 'StanfordCars', 'SUN397', 'UCF101']
INTERVALS = [256, 1, 4, 8, 16, 32, 64, 128]
SEEDS = [1, 2, 3]
SAVE_ROOT = 'viz/'


def main():
    if osp.exists(SAVE_ROOT):
        shutil.rmtree(SAVE_ROOT)
    os.makedirs(SAVE_ROOT)

    for interval in INTERVALS:
        print('Interval:', interval)
        all_plots_density, all_plots_point = [], []

        for dataset in DATASETS:
            plots_density, plots_point = [], []
            
            for seed in SEEDS:
                print(f'Stating on Dataset {dataset}, seed {seed}...')
                dfs = []

                for trainer in TRAINERS:
                    feats = read_feats(ROOT, trainer, dataset, seed)

                    n = max(feats['label']) + 1
                    m = math.ceil(n / 2)
                    base_labels = list(range(0, m))
                    novel_labels = list(range(m, n))
                    
                    base_feats = filter_labels(feats, base_labels)
                    novel_feats = filter_labels(feats, novel_labels)
                    
                    base_importances = cal_importance(base_feats, interval)
                    novel_importances = cal_importance(novel_feats, interval)

                    df = pd.DataFrame({
                        'channel_idx': range(len(base_importances)),
                        'base': base_importances,
                        'novel': novel_importances,
                        'trainer': trainer})
                    dfs.append(df.copy())
                
                df = pd.concat(dfs)
                df = df.replace({
                    'OracleStats': '1. Oracle',
                    'CoOpStats': '2. CoOp',
                    'ExtrasLinearProbeStats': '3. CoOp w/ Ours'})
                
                df['ratio'] = df['base'] / df['novel']

                p_density = (ggplot(df, aes('ratio', fill='trainer', color='trainer')) 
                    + geom_density(alpha=0.5)
                    + ggtitle(f'{dataset} seed{seed}')
                    + theme_seaborn()
                    + theme(axis_text_x=element_text(angle=0),
                            axis_text_y=element_text(angle=90), 
                            plot_title=element_text(hjust=0.5),
                            panel_spacing_x=0.02,
                            legend_position='bottom',
                            axis_title_x=element_text(face='bold'),
                            axis_title_y=element_text(face='bold'),
                            strip_background=element_blank(),
                            strip_text_x=element_blank())
                    + labs(x='Ratio', y='Density', color=''))

                plots_density.append(pw.load_ggplot(p_density))

                p_point = []
                for idx, df in enumerate(dfs):
                    trainer = df['trainer'].tolist()[0]
                    if idx == 1:
                        title = f'{dataset} seed{seed}\n{trainer}'
                    else:
                        title = trainer

                    df = df.sort_values('base')
                    df['order'] = range(len(df))
                    df = df.sort_values('channel_idx')
                    
                    df = pd.melt(df, id_vars=['channel_idx', 'trainer', 'order'], value_vars=['base', 'novel'])

                    p = (ggplot(df, aes('reorder(channel_idx, order)', 'value', color='variable'))
                        + geom_point()
                        + ggtitle(title)
                        + theme_seaborn()
                        + theme(axis_text_x=element_blank(),
                                plot_title=element_text(hjust=0.5),
                                legend_position='bottom',
                                axis_title_x=element_text(face='bold'),
                                axis_title_y=element_text(face='bold'))
                        + labs(x='Channel', y='Importance', color=''))

                    p_point.append(pw.load_ggplot(p, figsize=(4, 6)))

                p_point = pw.stack(p_point, operator='|')
                plots_point.append(p_point)

            all_plots_density.append(pw.stack(plots_density, operator='|'))
            all_plots_point.append(pw.stack(plots_point, operator='|'))

        p_density = pw.stack(all_plots_density, operator='/')
        p_point = pw.stack(all_plots_point, operator='/')
        p_density.savefig(f'{SAVE_ROOT}/all_density_channel{interval}.jpg')
        p_point.savefig(f'{SAVE_ROOT}/all_point_channel{interval}.jpg')


DATASETS = ['FGVCAircraft', 'EuroSAT']
SEEDS = [3, 2]
INTERVAL_POINT = 16
INTERVAL_DENSITY = 128
SAVE_ROOT = 'viz/'

TRAINER_TO_ALIAS = {
    'OracleStats': 'Oracle',
    'CoOpStats': 'CoOp',
    'ExtrasLinearProbeCoOpStats': 'w/ Ours'
}
TRAINER_TO_ALIAS_EUROSAT = {
    'OracleStats': 'Oracle (H=88.99)',
    'CoOpStats': 'CoOp (H=69.82)',
    'ExtrasLinearProbeCoOpStats': 'w/ Ours (H=75.70)'
}
TRAINER_TO_ALIAS_AIRCRAFT = {
    'OracleStats': 'Oracle (H=42.94)',
    'CoOpStats': 'CoOp (H=22.38)',
    'ExtrasLinearProbeCoOpStats': 'w/ Ours (H=29.46)'
}


def main2():
    if osp.exists(SAVE_ROOT):
        shutil.rmtree(SAVE_ROOT)
    os.makedirs(SAVE_ROOT)

    plots = []

    for dataset, seed in zip(DATASETS, SEEDS):
        print(f'Stating on Dataset {dataset}, seed {seed}...')
        dfs_point, dfs_density = [], []
        
        for trainer in TRAINERS:
            feats = read_feats(ROOT, trainer, dataset, seed)
            n = max(feats['label']) + 1
            m = math.ceil(n / 2)
            base_labels = list(range(0, m))
            novel_labels = list(range(m, n))
            
            base_feats = filter_labels(feats, base_labels)
            novel_feats = filter_labels(feats, novel_labels)

            base_importances_point = cal_importance(base_feats, INTERVAL_POINT)
            novel_importances_point = cal_importance(novel_feats, INTERVAL_POINT)        

            INTERVAL_DENSITY = 128 if dataset == 'EuroSAT' else 32
            base_importances_density = cal_importance(base_feats, INTERVAL_DENSITY)
            novel_importances_density = cal_importance(novel_feats, INTERVAL_DENSITY)

            df_point = pd.DataFrame({
                'channel_idx': range(len(base_importances_point)),
                'trainer': TRAINER_TO_ALIAS[trainer],
                'Base': base_importances_point,
                'Novel': novel_importances_point})
            df_density = pd.DataFrame({
                'channel_idx': range(len(base_importances_density)),
                'trainer': TRAINER_TO_ALIAS_EUROSAT[trainer] if dataset == 'EuroSAT' else TRAINER_TO_ALIAS_AIRCRAFT[trainer],
                'Base': base_importances_density,
                'Novel': novel_importances_density})
            
            dfs_point.append(df_point)
            dfs_density.append(df_density)

        df = pd.concat(dfs_point)
        values = df['Base'].tolist() + df['Novel'].tolist()
        ymin, ymax = min(values), max(values)

        p_point = []
        for idx, df in enumerate(dfs_point):
            df = df.sort_values('Base')
            df['order'] = range(len(df))
            df['order'] += 1
            df = df.sort_values('channel_idx')
            
            df = pd.melt(df, id_vars=['channel_idx', 'trainer', 'order'], 
                            value_vars=['Base', 'Novel'])

            trainer = df['trainer'].tolist()[0]

            legend_x = 0.62 if idx == 0 else 0.47
            p = (ggplot(df, aes('order', 'value', color='variable'))
                + geom_point(size=0.8)
                + annotate('text', x=len(df) / 4, y=0.95 * ymax, label=trainer)
                + labs(x='Channel' if idx == 1 else ' ', y='Chan. Import. (CI)' if idx == 0 else '', color='')
                + theme_seaborn()
                + scale_x_continuous(breaks=(1, 256, 512), expand=(0, 0))
                + scale_y_continuous(limits=(ymin, ymax), expand=(0, 0))
                + theme(axis_text_y=element_text() if idx == 0 else element_blank(),
                        axis_title_x=element_text(size=12),
                        axis_title_y=element_text(size=12),
                        legend_position=(legend_x, 0.25),
                        legend_direction='horizontal',
                        legend_text=element_text(size=8),
                        legend_key_size=6,
                        )
                )
            
            p_point.append(pw.load_ggplot(p, figsize=(1, 2)))

        p_point = pw.stack(p_point, operator='|', margin=0)
        p_point.set_suptitle(dataset, x=0.54, y=0.97, fontdict=dict(fontsize=12))
        plots.append(p_point)

        df = pd.concat(dfs_density)
        df['ratio'] = df['Base'] / df['Novel']

        p_density = (ggplot(df, aes('ratio', fill='trainer')) 
            + geom_density(alpha=0.5, color='none')
            + scale_x_continuous(expand=(0, 0))
            + scale_y_continuous(expand=(0, 0))
            + scale_fill_discrete(limits=list(TRAINER_TO_ALIAS_EUROSAT.values() if dataset == 'EuroSAT' 
                                              else TRAINER_TO_ALIAS_AIRCRAFT.values()))
            + theme_seaborn()
            + labs(x='CI-Base : CI-Novel', y='Frequency', fill='')
            + theme(axis_title_x=element_text(size=12),
                    axis_title_y=element_text(size=12),
                    legend_position=(0.72, 0.85),
                    legend_direction='vertical',
                    legend_text=element_text(size=8),
                    legend_key_size=6,
                    )
            )
        
        p_density = pw.load_ggplot(p_density, figsize=(2, 2))
        p_density.set_title(dataset, x=0.5, y=1.0, fontdict=dict(fontsize=12))
        plots.append(p_density)
    
    titles = ['(a)', '(b)', '(c)', '(d)']
    for title, plot in zip(titles, plots):
        if isinstance(plot, pw.Bricks):
            plot.set_supxlabel(title, y=-0.1, fontdict=dict(fontsize=16))
        else:
            plot.add_text(0.5, -0.11, title, fontdict=dict(fontsize=16))

    # p = pw.stack(plots, operator='|', margin=0.1)
    p1 = pw.stack([plots[0], plots[1]], operator='|', margin=0)
    p2 = pw.stack([plots[2], plots[3]], operator='|', margin=0)
    p = pw.stack([p1, p2], operator='|', margin=0.1)
    p.savefig(f'{SAVE_ROOT}/point_and_density.jpg')


if __name__ == '__main__':
    # main()
    main2()
