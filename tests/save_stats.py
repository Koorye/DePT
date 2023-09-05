import argparse
import pickle
import math
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.engine import build_trainer

import sys
sys.path.append('.')

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

import trainers.coop_stats
import trainers.elp_coop_stats
import trainers.oracle_stats
import trainers.zsclip_stats

from train import print_args, setup_cfg


@torch.no_grad()
def main(args):
    cfg = setup_cfg(args)
    print(cfg)
    
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    trainer.load_model(args.model_dir, epoch=args.load_epoch)
    trainer.set_model_mode("eval")

    split = cfg.TEST.SPLIT
    if split == "val" and trainer.val_loader is not None:
        data_loader = trainer.val_loader
    else:
        split = "test"  # in case val_loader is None
        data_loader = trainer.test_loader
    print(f"Evaluate on the *{split}* set")

    feats = defaultdict(list)
    n = trainer.dm.num_classes
    print('num classes:', n)
    labels = range(n)
    m = math.ceil(n / 2)
    base_labels = labels[:m]

    for batch_idx, batch in enumerate(tqdm(data_loader)):
        input_, labels = trainer.parse_batch_test(batch)

        trainer.is_base = labels[0] in base_labels
        text_feats, img_feats = trainer.model_inference(input_)

        labels = labels.detach().cpu().numpy()
        text_feats = text_feats.detach().cpu().numpy()
        img_feats = img_feats.detach().cpu().numpy()

        feats['label'].append(labels)
        feats['text'].append(text_feats)
        feats['img'].append(img_feats)
        
    feats['label'] = np.concatenate(feats['label'])
    feats['text'] = feats['text'][0]
    feats['img'] = np.concatenate(feats['img'])
    
    filename = f'feats_{args.trainer}_{cfg.DATASET.NAME}_seed{cfg.SEED}.pkl'
    with open(f'stats/{filename}', 'wb') as f:
        pickle.dump(feats, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    main(args)

