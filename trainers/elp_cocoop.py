# see elp_coopy.py to view comments
import os
import os.path as osp
import torch
import torch.nn.functional as F
from dassl.engine import TRAINER_REGISTRY
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import  build_lr_scheduler
from plotnine import *
from torch.cuda.amp import GradScaler, autocast
from torch import nn

from .cocoop import CoCoOp, load_clip_to_cpu
from .cocoop import CustomCLIP as CustomCLIP_
from .elp_maple import FiLM
from .optim import build_optimizer


class CustomCLIP(CustomCLIP_):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.subsample_classes = cfg.DATASET.SUBSAMPLE_CLASSES
        self.dataset = cfg.DATASET.NAME
        self.lp_cfg = cfg.TRAINER.LINEAR_PROBE
        self.film_cfg = cfg.TRAINER.FILM

        clip_dim = clip_model.text_projection.size(1)
        
        film_cfg = self.film_cfg

        if film_cfg.LINEAR_PROBE:
            self.film_lp_img = FiLM(clip_dim)
            self.film_lp_text = FiLM(clip_dim)
        
        if (self.subsample_classes == 'base') \
        or (self.subsample_classes == 'all' and 'ImageNet' in self.dataset):
            assert self.lp_cfg.TYPE in ['similarity', 'linear']

            if self.lp_cfg.TYPE == 'similarity':
                self.linear_probe_proj = nn.Identity()
            elif self.lp_cfg.TYPE == 'linear':
                self.linear_probe_proj = nn.Linear(clip_dim, len(classnames)).type(self.dtype)
        else:
            self.linear_probe_proj = nn.Identity()
        
    def forward(self, img, labels=None):
        if (self.subsample_classes == 'base') \
        or (self.subsample_classes == 'all' and 'ImageNet' in self.dataset):
            return self._forward_base(img, labels)
        else:
            return self._forward_new(img)

    def _forward_base(self, img, labels=None):
        text_feats_list, img_feats = self._forward_feats(img)
        
        logits = self._forward_logits_similarity(text_feats_list, img_feats)
        logits_lp, labels_lp = self._forward_logits_linear_probe(text_feats_list, img_feats, labels)
        
        if self.prompt_learner.training:
            return self._loss(logits, labels, logits_lp, labels_lp)
        
        if not self.lp_cfg.TEST_TIME_FUSION:
            return logits_lp

        lp_weight = self.lp_cfg.WEIGHT
        logits = (1 - lp_weight) * logits + lp_weight * logits_lp
        return logits
    
    def _forward_new(self, img):
        assert not self.prompt_learner.training
        
        text_feats, img_feats = self._forward_feats(img)
        logits = self._forward_logits_similarity(text_feats, img_feats)
        return logits
    
    def _forward_feats(self, img):
        img_feats = self.image_encoder(img.type(self.dtype))
        img_feats_norm = img_feats / img_feats.norm(dim=-1, keepdim=True)

        prompts = self.prompt_learner(img_feats_norm)

        tokenized_prompts = self.tokenized_prompts
        text_feats_list = [self.text_encoder(prompt, tokenized_prompts) for prompt in prompts]
        
        return text_feats_list, img_feats
    
    def _forward_logits_similarity(self, text_feats_list, img_feats):
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

        logits = []
        logit_scale = self.logit_scale.exp()

        for text_feats, img_feats_per_batch in zip(text_feats_list, img_feats):
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            logits_per_batch = logit_scale * img_feats_per_batch @ text_feats.t()
            logits.append(logits_per_batch)

        logits = torch.stack(logits)
        return logits
    
    def _forward_logits_linear_probe(self, text_feats_list, img_feats, labels):
        if self.film_cfg.LINEAR_PROBE:
            text_feats_list = [self.film_lp_text(text_feats) for text_feats in text_feats_list]
            img_feats = self.film_lp_img(img_feats)

        if self.lp_cfg.TYPE == 'similarity':
            return self._forward_logits_similarity(text_feats_list, img_feats)

        if labels is None:
            all_feats = img_feats
            all_labels = labels
        else:
            text_feats = torch.stack([text_feats[label] for text_feats, label 
                                      in zip(text_feats_list, labels)])
            all_feats = torch.cat([text_feats, img_feats])
            all_labels = torch.cat([labels, labels])

        all_logits = self.linear_probe_proj(all_feats)
        return all_logits, all_labels
    
    def _loss(self, logits, labels, logits_lp, labels_lp):
        loss_cls = F.cross_entropy(logits, labels)
        loss_cls_lp = F.cross_entropy(logits_lp, labels_lp)

        lp_weight = self.lp_cfg.WEIGHT
        loss = (1 - lp_weight) * loss_cls + lp_weight * loss_cls_lp
        return loss
    

@TRAINER_REGISTRY.register()
class ExtrasLinearProbeCoCoOp(CoCoOp):
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.COCOOP.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.COCOOP.PREC == "fp32" or cfg.TRAINER.COCOOP.PREC == "amp":
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        names_to_update = cfg.TRAINER.NAMES_TO_UPDATE

        for name, param in self.model.named_parameters():
            update = False

            for name_to_update in names_to_update:
                if name_to_update in name:
                    update = True
                    break
                
            param.requires_grad_(update)

        enabled = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.append(name)
        print(f"Parameters to be updated: {list(sorted(enabled))}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.optim, infos = build_optimizer(self.model, cfg.OPTIM)

        if infos is not None:
            print('Learning rate of parameters:')
            for info in infos:
                print('lr: {}, layers: {}'.format(info['lr'], info['layers']))
        
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("PromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COCOOP.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            if epoch < 0:
                all_model_files = os.listdir(osp.join(directory, name))
                all_model_files = [file_ for file_ in all_model_files if file_ != 'checkpoint']
                model_epochs = [int(file_.split('-')[-1]) for file_ in all_model_files]
                last_epoch = max(model_epochs)
                model_file = 'model.pth.tar-' + str(last_epoch)

            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))

            if self.cfg.DATASET.NAME in ['ImageNetA', 'ImageNetR']:
                from datasets.imagenet import ImageNet
                from dassl.utils import listdir_nohidden
                dataset = self.dm.dataset
                text_file = osp.join(dataset.dataset_dir, "classnames.txt")
                all_folders = ImageNet.read_classnames(text_file).keys()

                TO_BE_IGNORED = ["README.txt"]
                folders = listdir_nohidden(dataset.image_dir, sort=True)
                folders = [f for f in folders if f not in TO_BE_IGNORED]
                
                is_reserves = [f in folders for f in all_folders]

                print(f'State dict is CLIPPED to match the shape of target dataset {self.cfg.DATASET.NAME}!')
                state_dict['linear_probe_proj.weight'] = state_dict['linear_probe_proj.weight'][is_reserves]
                state_dict['linear_probe_proj.bias'] = state_dict['linear_probe_proj.bias'][is_reserves]
                
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
