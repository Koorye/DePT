import argparse
import os
import os.path as osp
import pandas as pd
import re


ORDERS_BASE_TO_NEW = ['imagenet', 'caltech101', 'oxford_pets', 'stanford_cars', 'oxford_flowers', 
                        'food101', 'fgvc_aircraft', 'sun397', 'dtd', 'eurosat', 'ucf101']

ORDERS_CROSS_DATASET = ['imagenet', 'caltech101', 'oxford_pets', 'stanford_cars', 'oxford_flowers', 
                          'food101', 'fgvc_aircraft', 'sun397', 'dtd', 'eurosat', 'ucf101', 
                          'imagenetv2', 'imagenet_sketch', 'imagenet_a', 'imagenet_r',]


class ResultParser(object):
    def __init__(self, mode, dir_, save_path):
        self.mode = mode
        self.dir_ = dir_
        self.save_path = save_path
    
    def parse_and_save(self):
        if self.mode == 'b2n':
            self.read_accs_base_to_new()
        elif self.mode == 'xd':
            self.read_accs_cross_dataset()
        
        self.save()

    def load_property(self, dir_):
        """ get property (trainer, datasets, num_shots, cfg, seeds) from directory """
        trainer = [subdir for subdir in os.listdir(dir_) if osp.isdir(osp.join(dir_, subdir))][0]
        
        dir_ = osp.join(dir_, trainer)
        datasets = os.listdir(dir_)

        if self.mode == 'b2n':
            datasets = [dataset for dataset in ORDERS_BASE_TO_NEW if dataset in datasets]
        elif self.mode == 'xd':
            datasets = [dataset for dataset in ORDERS_CROSS_DATASET if dataset in datasets]
        else:
            raise NotImplementedError
        
        dir_ = osp.join(dir_, datasets[0])
        num_shots = int(os.listdir(dir_)[0][5:])
        
        dir_ = osp.join(dir_, f'shots{num_shots}')
        cfg = os.listdir(dir_)[0]
        
        dir_ = osp.join(dir_, cfg)
        seeds = list(sorted([int(name[4:]) for name in os.listdir(dir_)]))
        
        self.prop = dict(
            trainer=trainer,
            datasets=datasets,
            num_shots=num_shots,
            cfg=cfg,
            seeds=seeds)

    def read_accs_base_to_new(self):
        dir_ = self.dir_

        base_dir = osp.join(dir_, 'train_base')
        new_dir = osp.join(dir_, 'test_new')
        
        self.load_property(base_dir)
        prop = self.prop

        trainer = prop['trainer']
        datasets = prop['datasets']
        num_shots = prop['num_shots']
        cfg = prop['cfg']
        seeds = prop['seeds']

        headers = ['dataset', 
                   'base_acc_seed1', 'new_acc_seed1', 'H_seed1', 
                   'base_acc_seed2', 'new_acc_seed2', 'H_seed2',
                   'base_acc_seed3', 'new_acc_seed3', 'H_seed3']
        rows = []
        
        for dataset in datasets:
            row = [dataset]
            
            for seed in seeds:
                base_path = osp.join(base_dir, trainer, dataset, f'shots{num_shots}', cfg, f'seed{seed}', 'log.txt')
                new_path = osp.join(new_dir, trainer, dataset, f'shots{num_shots}', cfg, f'seed{seed}', 'log.txt')

                base_acc = self._read_acc(base_path)
                new_acc = self._read_acc(new_path)
                H = 2 / (1 / base_acc + 1 / new_acc)
                
                row += [base_acc, new_acc, H]
            
            rows.append(row)
        
        df = pd.DataFrame(rows, columns=headers)
        df['base_acc'] = (df['base_acc_seed1'] + df['base_acc_seed2'] + df['base_acc_seed3']) / 3
        df['new_acc'] = (df['new_acc_seed1'] + df['new_acc_seed2'] + df['new_acc_seed3']) / 3

        df.loc[len(df.index)] = ['average'] + df.drop(columns=['dataset']).mean().tolist()
        df['H'] = 2 / (1 / df['base_acc'] + 1 / df['new_acc'])
        
        self.df = df


    def read_accs_cross_dataset(self):
        dir_ = self.dir_

        self.load_property(dir_)
        prop = self.prop

        trainer = prop['trainer']
        datasets = prop['datasets']
        num_shots = prop['num_shots']
        cfg = prop['cfg']
        seeds = prop['seeds']

        headers = ['dataset', 'acc_seed1', 'acc_seed2', 'acc_seed3']
        rows = []
        
        datasets = [dataset for dataset in ORDERS_CROSS_DATASET if dataset in datasets]
        for dataset in datasets:
            row = [dataset]
            
            for seed in seeds:
                path = osp.join(dir_, trainer, dataset, f'shots{num_shots}', cfg, f'seed{seed}', 'log.txt')
                acc = self._read_acc(path)
                row.append(acc)
            
            rows.append(row)
        
        df = pd.DataFrame(rows, columns=headers)
        df['acc'] = (df['acc_seed1'] + df['acc_seed2'] + df['acc_seed3']) / 3

        dg_datasets = [dataset for dataset in datasets
                      if 'imagenet' in dataset and dataset != 'imagenet']
        xd_datasets = [dataset for dataset in datasets
                      if dataset not in dg_datasets and dataset != 'imagenet']
        
        dg_df = df.loc[df['dataset'].isin(dg_datasets)].copy().reset_index(drop=True)
        xd_df = df.loc[df['dataset'].isin(xd_datasets)].copy().reset_index(drop=True)
        img_net_df = df.loc[df['dataset'] == 'imagenet'].copy().reset_index(drop=True)

        dg_df.loc[len(dg_df.index)] = ['average_dg'] + dg_df.drop(columns=['dataset']).mean().tolist()
        xd_df.loc[len(xd_df.index)] = ['average_xd'] + xd_df.drop(columns=['dataset']).mean().tolist()

        df = pd.concat([img_net_df, xd_df, dg_df]).reset_index(drop=True)
        
        self.df = df
        
    def save(self):
        save_path = self.save_path
        save_dir = osp.join(*save_path.replace('\\', '/').split('/')[:-1])

        os.makedirs(save_dir, exist_ok=True)
        self.df.round(2).to_csv(save_path, index=None)

    def _read_acc(self, path):
        with open(path, encoding='utf-8') as f:
            content = ''.join(f.readlines())
        try:
            acc = float(re.findall(r'accuracy\: (\d+\.\d*)\%', content)[-1])
            return acc
        except BaseException as e:
            print(f'Key word "accuracy" not found in file {path}!')
            raise e
    

def main(args):
    parser = ResultParser(args.mode, args.dir, args.save_path)
    parser.parse_and_save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help='mode, b2n or xd')
    parser.add_argument('--dir', type=str, help='directory which need to stats')
    parser.add_argument('--save-path', type=str, help='directory to save statistics')
    args = parser.parse_args()
    main(args)
