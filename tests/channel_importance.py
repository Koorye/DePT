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

def cal_importance(feats):
    importances = []
    for channel_ind_start in tqdm(range(0, 512)):
        channel_ind_end = channel_ind_start + 64
        channel_inds = list(range(channel_ind_start, channel_ind_end))
        channel_inds = [ind - 512 if ind >= 512 else ind for ind in channel_inds]
        importance = cal_importance_per_channel(feats, channel_inds)
        importances.append(importance)
        
    return np.array(importances)


ROOT = 'stats/'
TRAINERS = ['OracleStats', 'CoOpStats']
DATASETS = ['EuroSAT']
SEEDS = [1, 2, 3]
SAVE_ROOT = 'viz/'


def main():
    if osp.exists(SAVE_ROOT):
        shutil.rmtree(SAVE_ROOT)
    os.makedirs(SAVE_ROOT)

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
                
                base_importances = cal_importance(base_feats)
                novel_importances = cal_importance(novel_feats)

                df = pd.DataFrame({
                    'channel_idx': range(len(base_importances)),
                    'base': base_importances,
                    'novel': novel_importances,
                    'trainer': trainer})
                dfs.append(df.copy())
            
            df = pd.concat(dfs)
            
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
    p_density.savefig(f'{SAVE_ROOT}/all_density_channel.jpg')
    p_point.savefig(f'{SAVE_ROOT}/all_point_channel.jpg')


if __name__ == '__main__':
    main()
